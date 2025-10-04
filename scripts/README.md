# Unsloth GPT Training Scripts

## Production Scripts

### train.py - Main Training Script (v3)
**The primary training script with loss monitoring and all optimizations**

```bash
# Standard training
python train.py --profile standard --gpu 1

# With custom target loss
python train.py --profile standard --gpu 1 --target_loss 0.5
```

**Features:**
- Real-time loss monitoring with target tracking
- Overfitting/underfitting warnings
- Quality assessment on completion
- All optimizations: QLoRA, RSLoRA, cosine scheduler
- GPU selection before imports
- Organized model saving to `models/` folder

### train_standard.py - Standard Training Script (v2)
**Stable training script without loss monitoring**

```bash
python train_standard.py --profile standard --gpu 1
```

**Features:**
- Fixed GPU selection
- Organized model saving
- Consistent model naming
- Creates symlink to latest model

## Available Training Profiles

| Profile | Steps | Dataset | LoRA Rank | Use Case |
|---------|-------|---------|-----------|----------|
| quick_test | 30 | 100 | 16 | Testing configuration |
| standard | 200 | 1000 | 16 | Standard training |
| full | 500 | 5000 | 16 | Full training run |
| max_quality | 500 | 10000 | 16 | Maximum quality |
| conservative | 100 | 1000 | 8 | Low VRAM usage |

## Utility Scripts

- **chat.py** - Interactive chat interface
- **inference.py** - Batch inference
- **evaluate.py** - Model evaluation
- **benchmark.py** - Performance benchmarking
- **prepare_dataset.py** - Dataset preparation
- **export_to_llama.py** - Export to Llama format

## Archived Scripts

Legacy scripts have been moved to `archive/` folder:
- train_unsloth.py (original hardcoded script)
- train_simple.py
- train_advanced.py
- train_grpo.py
- train.py (first flexible version)

## Key Improvements in Production Scripts

1. **QLoRA Implementation**: Using 4-bit quantization with LoRA adapters
2. **GPU Selection Fix**: CUDA_VISIBLE_DEVICES set before imports
3. **Loss Monitoring**: Target loss of 0.5 for optimal generalization
4. **Cosine Scheduler**: Better convergence than linear
5. **Organized Output**: Models saved to `models/` with consistent naming
6. **Flexible Configuration**: Command-line arguments for all parameters

## Example Commands

```bash
# Quick test
python train.py --profile quick_test --gpu 1

# Standard training with custom learning rate
python train.py --profile standard --learning_rate 3e-4 --gpu 1

# Maximum quality training
python train.py --profile max_quality --target_loss 0.5 --gpu 1

# Conservative for low VRAM
python train.py --profile conservative --batch_size 2 --gpu 1
```

## Model Output Structure

```
models/
├── gpt-oss-20b_standard_cosine_r16_20251004_2115/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── training_info.txt
└── latest -> gpt-oss-20b_standard_cosine_r16_20251004_2115
```