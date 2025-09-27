# NeMo GPT-OSS-20B

Official NeMo implementation for OpenAI's GPT-OSS-20B model with Mixture of Experts (MoE) architecture.

## Project Status

✅ **PRODUCTION READY** - Ubuntu native setup complete
✅ **Model Downloaded** - GPT-OSS-20B ready (~13GB)
✅ **Docker Container** - Running with GPU access
✅ **Scripts Optimized** - OOM fixes applied for 24GB VRAM

## Model Specifications

- **Architecture**: Mixture of Experts (MoE) with 32 experts
- **Parameters**: 21B total, 3.6B active per token
- **Active Experts**: Configurable 1-32 (default: 2 for 24GB GPU)
- **Quantization**: MXFP4 for efficient memory usage
- **License**: Apache 2.0

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Start container
./start_nemo_container.sh

# Access container
docker exec -it nemo-gpt-oss bash

# Run inference
cd /workspace
python inference.py
```

### Option 2: Native Installation

```bash
# Install CUDA stack
./install_cuda_stack.sh
source ~/.bashrc

# Install Python ML stack
./install_python_ml_stack.sh
source venv/bin/activate

# Run inference
python workspace/inference.py
```

## Structure

```
.
├── models/
│   └── gpt-oss-20b/             # Downloaded model (~13GB)
├── workspace/
│   ├── inference.py             # Comprehensive inference script
│   ├── convert_to_nemo.py       # HF to NeMo conversion
│   ├── gpt_oss_training.py      # Training script
│   └── checkpoints/             # Model checkpoints
├── nemo/                        # NVIDIA NeMo repository
├── install_docker.sh            # Docker installation
├── install_nvidia_toolkit.sh    # NVIDIA Container Toolkit
├── install_cuda_stack.sh        # CUDA/cuDNN installation
├── install_python_ml_stack.sh   # Python ML environment
├── start_nemo_container.sh      # Container startup
├── UBUNTU_SETUP_GUIDE.md        # Detailed Ubuntu setup guide
└── README.md                    # This file
```

## System Specifications

- **OS**: Ubuntu 24.04 (native)
- **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
- **Driver**: 580.65.06 (CUDA 13.0 support)
- **Disk**: 5TB NVMe SSD (4.4TB free)
- **Container**: nvcr.io/nvidia/nemo:25.07.gpt_oss (36.4GB)

## Key Features

### Inference Script (workspace/inference.py)
- 100+ configurable parameters with documentation
- MoE settings (topk, load balancing, routing)
- Parallelism options (tensor, pipeline, expert)
- Precision control (bf16, fp16, fp32, fp8)
- Generation parameters (temperature, top-p, top-k)
- OOM optimizations for 24GB VRAM

### Optimizations Applied
- `MOE_TOPK`: 2 (reduced from 4)
- `MAX_BATCH_SIZE`: 1 (reduced from 8)
- `USE_CPU_INITIALIZATION`: True (critical for 24GB)
- `PYTORCH_CUDA_ALLOC_CONF`: expandable_segments enabled

## Installation Scripts

All installation scripts are ready to use:

1. **install_docker.sh** - Install Docker CE
2. **install_nvidia_toolkit.sh** - Install NVIDIA Container Toolkit
3. **install_cuda_stack.sh** - Install CUDA 12.6 + cuDNN + NVIDIA libs
4. **install_python_ml_stack.sh** - Install PyTorch, NeMo, Transformers, etc.
5. **start_nemo_container.sh** - Start NeMo container with GPU

See [UBUNTU_SETUP_GUIDE.md](UBUNTU_SETUP_GUIDE.md) for detailed instructions.

## Usage

### Docker Container Commands

```bash
# Start container
./start_nemo_container.sh

# Access container
docker exec -it nemo-gpt-oss bash

# Check GPU
docker exec nemo-gpt-oss nvidia-smi

# Stop container
docker stop nemo-gpt-oss

# View logs
docker logs nemo-gpt-oss
```

### Run Inference

Inside container or native environment:

```bash
cd /workspace  # Or workspace/ if native
python inference.py
```

Customize prompts in `inference.py`:
```python
PROMPTS = [
    "Q: What is artificial intelligence?",
    "Explain quantum computing in simple terms:",
    "Write a short story about a robot:",
]
```

## Configuration

Key parameters in `workspace/inference.py`:

```python
MOE_TOPK = 2              # Number of active experts (1-32)
TEMPERATURE = 0.7         # Randomness (0.0-2.0)
TOP_P = 0.9              # Nucleus sampling
NUM_TOKENS_TO_GENERATE = 100
MAX_BATCH_SIZE = 1        # Prompts per batch
PRECISION = "bf16-mixed"  # bf16/fp16/fp32
```

## Troubleshooting

### Out of Memory Errors
- Reduce `MOE_TOPK` to 1
- Reduce `NUM_TOKENS_TO_GENERATE`
- Set `MAX_BATCH_SIZE = 1`

### Container Won't Start
```bash
sudo systemctl restart docker
docker ps -a
docker logs nemo-gpt-oss
```

### GPU Not Accessible
```bash
nvidia-smi
docker exec nemo-gpt-oss nvidia-smi
```

## Documentation

- [UBUNTU_SETUP_GUIDE.md](UBUNTU_SETUP_GUIDE.md) - Complete Ubuntu setup
- [SETUP_STATUS.md](SETUP_STATUS.md) - Legacy Windows/WSL setup
- [CUDA_COMPATIBILITY_NOTES.md](CUDA_COMPATIBILITY_NOTES.md) - CUDA compatibility

## References

- NeMo Framework: https://github.com/NVIDIA/NeMo
- Documentation: https://docs.nvidia.com/nemo-framework/
- GPT-OSS Model: https://huggingface.co/openai/gpt-oss-20b
- OpenAI GPT-OSS: https://github.com/openai/gpt-oss

## License

Apache 2.0

---

**Last Updated**: September 27, 2025
**Status**: Production Ready on Ubuntu 24.04