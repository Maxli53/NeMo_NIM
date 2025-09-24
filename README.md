# NeMo GPT-OSS-20B

Official NeMo implementation following NVIDIA documentation.

## Structure

```
.
├── nemo/                    # Official NVIDIA NeMo repository (cloned from GitHub)
└── workspace/              # Working directory
    ├── checkpoints/        # Model checkpoints
    └── gpt_oss_training.py # Training script (from official docs)
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
docker run --gpus all -it \
  -v $(pwd)/workspace:/workspace \
  nvcr.io/nvidia/nemo:25.07.gpt_oss
```

## Usage

Inside the container, follow the official documentation:
- https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/gpt_oss.html

## References

- NeMo Framework: https://github.com/NVIDIA/NeMo
- Documentation: https://docs.nvidia.com/nemo-framework/