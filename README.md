# vLLM with sm_70 (Volta) Support

Docker image for running **vLLM v0.12.0** on older NVIDIA GPUs with **sm_70** compute capability (Volta architecture), including:
- Tesla V100
- Titan V
- Quadro GV100
- **NVIDIA CMP 100-210** (mining GPUs)

## Why This Exists

Official vLLM Docker images don't support sm_70 GPUs. This project builds PyTorch 2.9.0 and vLLM 0.12.0 from source with `TORCH_CUDA_ARCH_LIST="7.0"` to enable Volta support.

## Pre-built Image

Pull the pre-built image from GitHub Container Registry:

```bash
docker pull ghcr.io/jajmangold/vllm-sm70:latest
```

Available tags:
- `ghcr.io/jajmangold/vllm-sm70:latest` - Latest build
- `ghcr.io/jajmangold/vllm-sm70:0.12.0` - vLLM 0.12.0

## Quick Start

### Single GPU

```bash
# Pull and run with a model
docker run --gpus all -p 8000:8000 ghcr.io/jajmangold/vllm-sm70:latest \
  --model Qwen/Qwen3-4B-Instruct \
  --dtype half \
  --trust-remote-code

# Or use docker-compose (update image name first)
docker compose up
```

### Dual GPU with Load Balancing

The included `docker-compose.yml` sets up:
- **vLLM on GPU 0** (port 8000)
- **vLLM on GPU 1** (port 8001)
- **nginx load balancer** (port 80) with session affinity

```bash
docker compose up -d
```

Access the API:
- **Load balanced**: `http://localhost:80/v1/chat/completions`
- **GPU 0 direct**: `http://localhost:8000/v1/chat/completions`
- **GPU 1 direct**: `http://localhost:8001/v1/chat/completions`

**Session Affinity**: Pass `X-Session-ID` header to route same user to same backend for KV cache reuse:
```bash
curl -X POST http://localhost:80/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: user123" \
  -d '{"model": "...", "messages": [...]}'
```

### Build from Source (Optional)

If you prefer to build locally (~3-4 hours):

```bash
docker build -t vllm-sm70 .
```

## Performance Benchmarks

Tested on **2x NVIDIA CMP 100-210** (16GB each, PCIe x1) with `Qwen3-4B-Instruct-2507-GPTQ-Int4`:

### Text Generation (150 tokens)

| Setup | Concurrency | Throughput |
|------|-------------|------------|
| Single GPU | 40 | ~580 tok/s |
| Dual GPU | 60 (30+30) | **~1170 tok/s** |
| Single request | 1 | ~105 tok/s |

**Optimal Configuration:**
- `--max-model-len=512` (context length)
- `--max-num-batched-tokens=4096`
- `--max-num-seqs=128`
- `--gpu-memory-utilization=0.90`
- CUDA graphs enabled (no `--enforce-eager`)

### Logit-Only Decisions (1 token)

Perfect for classification, routing, and scoring:

| Setup | Concurrency | Throughput |
|------|-------------|------------|
| Dual GPU | 200 (100+100) | **~200 decisions/sec** |

**Use Cases:**
- Content moderation
- Intent classification
- Routing to specialized models
- Yes/no validation
- Sentiment scoring

### Key Optimizations

1. **CUDA Graphs**: 4x single-request latency improvement (26 → 105 tok/s)
2. **Short Context (512)**: Reduces KV cache pressure, enables higher concurrency
3. **Dual GPU**: ~2x throughput with proper load balancing
4. **Session Affinity**: KV cache reuse for multi-turn conversations

## Requirements

- NVIDIA GPU with sm_70 compute capability
- NVIDIA Driver 535+ (CUDA 12.x compatible)
- Docker with NVIDIA Container Toolkit
- ~50GB disk space for build
- ~40GB for final image

**PCIe Limitation**: CMP 100-210 cards typically have PCIe x1 slots. This limits bandwidth but doesn't prevent excellent performance for inference workloads. The dual-GPU setup achieves ~1170 tok/s despite this constraint.

## Configuration

Edit `docker-compose.yml` to customize:
- Model selection
- Tensor parallelism (for multi-GPU)
- Memory utilization
- Context length
- Batch sizes

**Recommended Settings:**
```yaml
- --max-model-len=512          # Context length
- --max-num-batched-tokens=4096 # Prefill batch budget
- --max-num-seqs=128           # Max concurrent sequences
- --gpu-memory-utilization=0.90 # Memory usage
- --swap-space=0               # Disable swap
```

## Architecture Notes

### What Works Well

- ✅ **CUDA graphs** - Major latency win
- ✅ **Triton attention** - Best option on Volta
- ✅ **Continuous batching** - Excellent scaling
- ✅ **GPTQ int4** - Good quality/speed tradeoff
- ✅ **Short context (256-512)** - Optimal for high throughput

### Limitations

- ❌ **Flash Attention 2** - Requires sm_80+ (Ampere)
- ❌ **fp8/int8 KV cache** - Requires sm_89+ (Ada/Hopper)
- ❌ **Marlin kernels** - Require sm_80+ (Ampere)
- ⚠️ **PCIe x1 bandwidth** - Can bottleneck at very high concurrency

### Why This Setup Works

For **short context, high concurrency** workloads:
- Attention cost is minimal (O(n²) with small n)
- KV cache fits easily in VRAM
- Continuous batching amortizes overhead
- PCIe bandwidth is sufficient for inference (not training)

This makes CMP cards excellent **control-plane LLM appliances**:
- Decision-heavy workloads
- High-throughput serving
- Cost-effective inference

## Build Details

- **Base**: `nvidia/cuda:12.8.0-devel-ubuntu24.04`
- **PyTorch**: 2.9.0 (built from source with sm_70)
- **vLLM**: 0.12.0
- **CUDA Arch**: 7.0 (Volta)

## Usage Examples

### Text Generation

```bash
curl -X POST http://localhost:80/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: user123" \
  -d '{
    "model": "JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4",
    "messages": [{"role": "user", "content": "Write a story about robots."}],
    "max_tokens": 150
  }'
```

### Logit-Only Decision

```bash
curl -X POST http://localhost:80/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4",
    "messages": [{"role": "user", "content": "Is this spam? Reply YES or NO: Buy cheap watches!"}],
    "max_tokens": 1
  }'
```

## Notes

- Flash Attention 2 is **not** supported on Volta GPUs - vLLM will use fallback attention
- Use `--enforce-eager` to disable CUDA graphs if you encounter issues (not recommended - major performance hit)
- Build takes 3-4 hours due to PyTorch compilation
- For best results, use quantized models (GPTQ int4) to fit in VRAM and maximize throughput

## License

MIT
