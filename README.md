# Unsloth GPT-OSS-20B Fine-tuning

Efficient fine-tuning of GPT-OSS-20B using Unsloth. Requires only 14GB VRAM on RTX 3090.

## Quick Start

```bash
# Setup (one-time)
./setup.sh
source venv/bin/activate

# Train (30 steps test)
python scripts/train_advanced.py --profile quick_test

# Inference
python scripts/inference.py --model_path unsloth/gpt-oss-20b --interactive
```

## Installation

### Automated
```bash
./setup.sh  # Installs everything with UV package manager
```

### Manual
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade -qqq uv
uv pip install -qqq \
    "torch>=2.8.0" "triton>=3.4.0" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth" \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    bitsandbytes transformers datasets trl
```

## Training

### Profiles
- `quick_test`: 30 steps validation
- `standard`: 100 steps training
- `high_quality`: r=32, 2 epochs
- `memory_efficient`: r=8, 1024 context

```bash
# Standard training
python scripts/train_advanced.py --profile standard --validate

# Custom configuration
python scripts/train_simple.py \
    --model_name unsloth/gpt-oss-20b \
    --lora_r 16 \
    --max_steps 100
```

### Key Settings (RTX 3090)
```yaml
Model: unsloth/gpt-oss-20b
QLoRA: 14GB VRAM
Batch: 2, Accumulation: 8 (effective=16)
LoRA rank: 16, Alpha: 16
Learning rate: 2e-4
Max epochs: 1-2
```

## Inference

```bash
# Interactive chat
python scripts/inference.py \
    --model_path ./final_model \
    --interactive \
    --reasoning_effort medium

# Benchmark
python scripts/benchmark.py --model_path unsloth/gpt-oss-20b
```

Settings: temp=1.0, top_p=1.0, top_k=0

## Export

```bash
# To GGUF for llama.cpp
python scripts/export_to_llama.py \
    --model_path ./final_model \
    --quantization Q4_K_M
```

## Project Structure

```
├── configs/training_optimal.yaml  # All training configs
├── scripts/
│   ├── train_simple.py           # Basic training
│   ├── train_advanced.py         # With monitoring
│   ├── inference.py              # Generation
│   ├── export_to_llama.py       # GGUF export
│   └── benchmark.py              # Performance test
├── models/gpt-oss-20b/           # Model storage
├── data/                         # Datasets
└── setup.sh                      # Setup script
```

## Hyperparameter Guide

### Avoiding Overfitting
- Loss < 0.2: Reduce epochs, lower LR, increase weight_decay
- Use r=16, alpha=16 (ratio=1)

### Best Practices
- Target all 7 modules (attention + MLP)
- Train on completions only (+1-3% accuracy)
- Effective batch size = 16
- QLoRA saves 75% VRAM vs LoRA

### Memory by Configuration
| Setting | VRAM | Speed |
|---------|------|-------|
| QLoRA r=8 | 14GB | Fast |
| QLoRA r=16 | 16GB | Balanced |
| LoRA r=16 | 44GB | Fastest |

## Resources

- [Unsloth Docs](https://docs.unsloth.ai/)
- [GPT-OSS Guide](https://docs.unsloth.ai/new/gpt-oss-how-to-run-and-fine-tune)
- [Model Hub](https://huggingface.co/unsloth/gpt-oss-20b-GGUF)

---
Hardware: 2x RTX 3090 (24GB each) | Created: 2025-09-29