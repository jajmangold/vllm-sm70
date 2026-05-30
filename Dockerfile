# vLLM with sm_70 (Volta) support
# Builds vLLM from source so its CUDA kernels include sm_70.
#
# IMPORTANT (Volta + torch): recent vLLM (0.20.x) pins torch==2.11.0, and torch
# 2.11 DROPPED Volta from its cu128/cu130 binaries. Only the **cu126** build of
# torch 2.11 still ships sm_70. Basing on a cu128/cu130 image therefore produces
# an image that dies on V100 at CUDA init with "no kernel image is available for
# execution on the device" (torch's own kernels lack sm_70), even though vLLM's
# kernels were compiled for 7.0. So we base on the cuda12.6 PyTorch image.
ARG PYTORCH_IMAGE=pytorch/pytorch:2.11.0-cuda12.6-cudnn9-devel
FROM ${PYTORCH_IMAGE}

ARG VLLM_VERSION=0.20.2
ARG MAX_JOBS=8
ARG NVCC_THREADS=2

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="7.0" \
    CMAKE_CUDA_ARCHITECTURES=70 \
    CUDAARCHS=70 \
    CMAKE_CUDA_FLAGS="-gencode=arch=compute_70,code=sm_70 -Wno-deprecated-gpu-targets" \
    NVCC_PREPEND_FLAGS="-gencode=arch=compute_70,code=sm_70 -Wno-deprecated-gpu-targets" \
    HF_HOME=/root/.cache/huggingface \
    NVIDIA_DISABLE_REQUIRE=1 \
    VLLM_TARGET_DEVICE=cuda \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    ninja-build \
    curl \
    ca-certificates \
    patchelf \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip "setuptools<82" wheel

RUN git clone --depth 1 --branch "v${VLLM_VERSION}" \
        https://github.com/vllm-project/vllm.git /opt/vllm

WORKDIR /opt/vllm

RUN python3 -c 'from pathlib import Path; cm=Path("CMakeLists.txt"); txt=cm.read_text(); old="    # vllm-flash-attn should be last as it overwrites some CMake functions\n    include(cmake/external_projects/vllm_flash_attn.cmake)\n"; new="    message(STATUS \"Skipping vllm-flash-attn for Volta-only sm_70 build\")\n"; assert old in txt, "vllm-flash-attn CMake include block not found"; cm.write_text(txt.replace(old, new)); setup=Path("setup.py"); txt=setup.read_text(); start="if _is_cuda():\n    ext_modules.append(CMakeExtension(name=\"vllm.vllm_flash_attn._vllm_fa2_C\"))"; end="    if envs.VLLM_USE_PRECOMPILED or (\n        CUDA_HOME and get_nvcc_cuda_version() >= Version(\"12.9\")\n    ):"; i=txt.index(start); j=txt.index(end, i); repl="if _is_cuda():\n    # Volta cannot use FA2/FA3/FA4 kernels; skip building those extensions.\n"; setup.write_text(txt[:i] + repl + txt[j:])'

RUN python3 -c 'from pathlib import Path; import torch, triton; Path("/opt/vllm/constraints.txt").write_text(f"torch=={torch.__version__}\ntriton=={triton.__version__}\n")'

RUN python3 use_existing_torch.py && \
    pip install --no-cache-dir -r requirements/build/cuda.txt && \
    SETUPTOOLS_SCM_PRETEND_VERSION=${VLLM_VERSION} \
    MAX_JOBS=${MAX_JOBS} NVCC_THREADS=${NVCC_THREADS} \
        pip install --no-cache-dir --no-build-isolation -c /opt/vllm/constraints.txt -v .

RUN pip install --no-cache-dir -c /opt/vllm/constraints.txt bitsandbytes auto-round

# --- Volta runtime patches -------------------------------------------------
# We deliberately don't build the FlashAttention extensions (no FA on sm_70).
# But two code paths still import them unconditionally and break model load for
# anything that pulls in MLA/mamba layers (e.g. Qwen3.5). Patch both to degrade
# gracefully to the non-FA / native path. (The published images never hit these
# because they die earlier on the torch/CUDA-arch mismatch fixed by the cu126 base.)

# 1) vllm.vllm_flash_attn hard-raises ImportError when the FA C-extensions are
#    absent. Don't raise: the symbols stay importable (FA2/FA3_AVAILABLE=False)
#    and vLLM selects a non-FlashAttention (Triton) backend at runtime.
RUN python3 - <<'PY'
import glob
from pathlib import Path
p = Path(glob.glob("/usr/local/lib/python3*/dist-packages/vllm/vllm_flash_attn/__init__.py")[0])
t = p.read_text()
old = (
    "if not (FA2_AVAILABLE or FA3_AVAILABLE):\n"
    "    raise ImportError(\n"
    '        "vllm.vllm_flash_attn requires the CUDA flash attention extensions "\n'
    '        "(_vllm_fa2_C or _vllm_fa3_C). On ROCm, use upstream flash_attn."\n'
    "    )"
)
assert old in t, "FA raise block not found"
t = t.replace(
    old,
    "# sm_70 (Volta) build: FA2/FA3 kernels are not built; do not raise.\n"
    "# vLLM selects a non-FlashAttention (Triton) backend at runtime.\n"
    "if False:\n"
    "    raise ImportError('unreachable')",
)
p.write_text(t)
print("patched", p)
PY

# 2) The rotary-embedding custom op's forward_cuda imports apply_rotary_emb from
#    vllm.vllm_flash_attn.layers.rotary (part of the FA submodule we don't build).
#    Fall back to the pure-torch forward_native when that import is missing
#    (used by Qwen3.5's vision tower; harmless elsewhere on Volta).
RUN python3 - <<'PY'
import glob
from pathlib import Path
p = Path(glob.glob("/usr/local/lib/python3*/dist-packages/vllm/model_executor/layers/rotary_embedding/common.py")[0])
t = p.read_text()
old = "        from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb\n"
new = (
    "        try:\n"
    "            from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb\n"
    "        except ModuleNotFoundError:\n"
    "            # Volta sm_70 build: FA rotary helper not built; use native path.\n"
    "            return self.forward_native(x, cos, sin)\n"
)
assert old in t, "rotary FA import line not found"
p.write_text(t.replace(old, new, 1))
print("patched rotary", p)
PY

WORKDIR /root

# Build-time sanity check. NOTE: sm_70 and `import vllm._C` CANNOT be validated
# here -- both need libcuda.so.1, which isn't present during `docker build` (no
# --gpus / no driver on CI runners). Calling torch.cuda.get_arch_list() without a
# GPU returns []. So we only assert the Volta-critical invariant: torch is a cu126
# build (the cu126 wheels are what include sm_70). sm_70 + a real forward pass are
# validated at container runtime, on the GPU.
RUN python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert "cu126" in torch.__version__, torch.__version__
PY

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
