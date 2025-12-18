# vLLM with sm_70 (Volta) Support

Docker image for running **vLLM v0.12.0** on older NVIDIA GPUs with **sm_70** compute capability (Volta architecture), including:
- Tesla V100
- Titan V
- Quadro GV100
- **NVIDIA CMP 100-210** (mining GPUs)

## Why This Exists

Official vLLM Docker images don't support sm_70 GPUs. This project builds PyTorch 2.9.0 and vLLM 0.12.0 from source with `TORCH_CUDA_ARCH_LIST="7.0"` to enable Volta support.

## Quick Start

```bash
# Build the image (~3-4 hours, builds PyTorch from source)
docker build -t vllm-sm70 .

# Run with a model
docker run --gpus all -p 8000:8000 vllm-sm70 \
  --model Qwen/Qwen3-4B-Instruct \
  --dtype half \
  --trust-remote-code

# Or use docker-compose
docker compose up
```

## Requirements

- NVIDIA GPU with sm_70 compute capability
- NVIDIA Driver 535+ (CUDA 12.x compatible)
- Docker with NVIDIA Container Toolkit
- ~50GB disk space for build
- ~40GB for final image

## Configuration

Edit `docker-compose.yml` to customize:
- Model selection
- Tensor parallelism (for multi-GPU)
- Memory utilization
- Context length

## Build Details

- **Base**: `nvidia/cuda:12.8.0-devel-ubuntu24.04`
- **PyTorch**: 2.9.0 (built from source with sm_70)
- **vLLM**: 0.12.0
- **CUDA Arch**: 7.0 (Volta)

## Notes

- Flash Attention 2 is **not** supported on Volta GPUs - vLLM will use fallback attention
- Use `--enforce-eager` to disable CUDA graphs if you encounter issues
- Build takes 3-4 hours due to PyTorch compilation

## License

MIT

