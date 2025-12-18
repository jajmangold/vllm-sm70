# vLLM v0.12.0 + PyTorch (source) + sm_70 (Volta / CMP 100-210)
# Tested design for CUDA-driver 13.x hosts

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="7.0" \
    CUDAARCHS="70" \
    MAX_JOBS=4 \
    USE_CUDA=1 \
    USE_CUDNN=1 \
    BUILD_TEST=0 \
    PYTORCH_BUILD_VERSION=2.9.0+sm70 \
    PYTORCH_BUILD_NUMBER=1 \
    HF_HOME=/root/.cache/huggingface

# ---- system deps ----
RUN apt update && apt install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    build-essential \
    cmake \
    ninja-build \
    curl \
    ca-certificates \
    libopenblas-dev \
    libomp-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install --break-system-packages numpy wheel "setuptools>=70" pyyaml typing_extensions

# ---- build PyTorch from source (sm_70 enabled) ----
WORKDIR /opt
RUN git clone --recursive https://github.com/pytorch/pytorch.git
WORKDIR /opt/pytorch
RUN git checkout v2.9.0 && \
    git submodule sync && \
    git submodule update --init --recursive

# Disable stuff we don't need (saves hours)
ENV USE_MKLDNN=0 \
    USE_FBGEMM=0 \
    USE_NNPACK=0 \
    USE_XNNPACK=0 \
    BUILD_CAFFE2=0 \
    USE_QNNPACK=0 \
    USE_ROCM=0

RUN pip install --break-system-packages -e . -v --no-build-isolation

# ---- build vLLM v0.12.0 ----
WORKDIR /opt
RUN git clone --recursive https://github.com/vllm-project/vllm.git
WORKDIR /opt/vllm
RUN git checkout v0.12.0 && git submodule update --init --recursive

# vLLM build deps
RUN python3 -m pip install --break-system-packages setuptools_scm cmake ninja packaging
RUN python3 -m pip install --break-system-packages --no-build-isolation -e .

# ---- runtime ----
EXPOSE 8000
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
