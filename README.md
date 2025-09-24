# NeMo GPT-OSS-20B Project

Professional implementation of OpenAI's GPT-OSS-20B using NVIDIA NeMo Framework and NIM deployment.

## Overview

This project provides a complete setup for:
- Training and fine-tuning GPT-OSS-20B with NeMo
- Deploying models using NVIDIA NIM
- Optimized inference on consumer GPUs (RTX 3090)

Based on official NVIDIA documentation:
- [NeMo Framework](https://docs.nvidia.com/nemo-framework/)
- [GPT-OSS Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/gpt_oss.html)
- [NeMo GitHub](https://github.com/NVIDIA/NeMo)

## Project Structure

```
nemo-gpt-oss/
├── workspace/          # Working directory for experiments
│   ├── experiments/    # NeMo-Run experiment configs
│   ├── checkpoints/    # Model checkpoints
│   ├── data/          # Training/evaluation data
│   └── outputs/       # Logs and results
├── scripts/           # Python scripts for training/inference
├── configs/           # Model and training configurations
├── nim/              # NIM deployment files
├── nemo/             # Cloned NeMo repository (reference)
└── requirements/     # Python dependencies
```

## Quick Start

```bash
# 1. Setup environment
./setup.sh

# 2. Import model to NeMo format
docker-compose run nemo-dev python /scripts/import_gpt_oss.py \
  --source /workspace/checkpoints/base/gpt-oss-20b

# 3. Fine-tune with LoRA
docker-compose run nemo-dev python /scripts/train_gpt_oss.py \
  --task finetune --peft-scheme lora

# 4. Run inference
docker-compose run nemo-dev python /scripts/generate.py \
  --model-path /workspace/checkpoints/converted/gpt_oss_20b.nemo \
  --prompt "Hello, world!"
```

## Documentation

See full documentation in the README for:
- Detailed setup instructions
- Configuration options
- Performance optimization
- API usage
- Troubleshooting

Based on official NVIDIA NeMo and NIM documentation.
