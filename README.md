# NeMo GPT-OSS-20B

Official NeMo implementation for OpenAI's GPT-OSS-20B model with Mixture of Experts (MoE) architecture.

## Project Status

✅ **Environment Setup Complete** - NeMo container running with GPU support
⚠️ **Awaiting Model Download** - Need 30GB disk space to proceed

## Model Specifications

- **Architecture**: Mixture of Experts (MoE) with 32 experts
- **Parameters**: 21B total, 3.6B active per token
- **Active Experts**: Configurable 1-32 (default: 4)
- **Quantization**: MXFP4 for 16GB memory usage
- **License**: Apache 2.0

## Structure

```
.
├── nemo/                        # Official NVIDIA NeMo repository (602MB)
├── workspace/                   # Working directory
│   ├── gpt_oss_training.py     # Training script with MoE config
│   └── test_gpt_oss_config.py  # Configuration testing
├── CUDA_COMPATIBILITY_NOTES.md # CUDA setup documentation
├── SETUP_STATUS.md             # Detailed setup status
└── README.md                   # This file
```

## Setup

1. **Install NVIDIA Container Toolkit** (in WSL2):
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

2. **Pull official container**:
```bash
docker pull nvcr.io/nvidia/nemo:25.07.gpt_oss
```

3. **Run container**:
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -it -v /mnt/c/Users/maxli/PycharmProjects/PythonProject/AI_agents/workspace:/workspace \
  --name nemo-gpt-oss -d nvcr.io/nvidia/nemo:25.07.gpt_oss bash
```

## Usage

Inside the container, follow the official documentation:
- https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/gpt_oss.html

## System Requirements

- **GPU**: NVIDIA RTX 3090 or better (24GB VRAM)
- **Driver**: 575+ for CUDA 12.9 support (current: 581.29)
- **Disk Space**: 30GB minimum for model download and conversion
- **Container Size**: 36.4GB for nvcr.io/nvidia/nemo:25.07.gpt_oss

## References

- NeMo Framework: https://github.com/NVIDIA/NeMo
- Documentation: https://docs.nvidia.com/nemo-framework/
- GPT-OSS Model: https://huggingface.co/openai/gpt-oss-20b
- OpenAI GPT-OSS: https://github.com/openai/gpt-oss