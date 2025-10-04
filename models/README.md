# Unsloth GPT Models Directory

## Directory Structure

```
models/
├── gpt-oss-20b_{profile}_{timestamp}/    # Trained models
├── latest -> [symlink to most recent]    # Latest model
├── archive/                              # Old/legacy models
└── .gitkeep                              # Git placeholder
```

## Naming Convention

All models follow this simplified pattern:
```
gpt-oss-20b_{profile}_{timestamp}
```

### Components:
- **gpt-oss-20b**: Base model identifier
- **{profile}**: Training profile (quick_test, standard, full, max_quality, conservative)
- **{timestamp}**: Format YYYYMMDD_HHMM

### Examples:
- `gpt-oss-20b_standard_20251004_2115`
- `gpt-oss-20b_max_quality_20251005_0930`
- `gpt-oss-20b_conservative_20251004_1430`

## Model Contents

Each model directory contains:
```
model_folder/
├── adapter_config.json           # LoRA configuration
├── adapter_model.safetensors     # LoRA weights
├── tokenizer_config.json         # Tokenizer config
├── tokenizer.json                # Tokenizer
├── training_info.txt             # Training details (includes rank, scheduler, loss)
└── checkpoint-*/                 # Training checkpoints
```

## Quality Indicators

Based on final loss:
- 🎯 **0.4-0.7**: Optimal
- ✅ **0.7-1.0**: Good
- ⚠️ **<0.3**: Overfitted
- 🔴 **>1.0**: Underfitted

## Loading Models

```python
from unsloth import FastLanguageModel
from peft import PeftModel

# Load latest
model_path = "models/latest"

# Or specific model
model_path = "models/gpt-oss-20b_standard_20251004_2115"

# Load base + adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, model_path)
FastLanguageModel.for_inference(model)
```

## Training Profiles

| Profile | Steps | Dataset | Use Case |
|---------|-------|---------|----------|
| quick_test | 30 | 100 | Testing |
| standard | 200 | 1000 | General |
| full | 500 | 5000 | Production |
| max_quality | 500 | 10000 | Best quality |
| conservative | 100 | 1000 | Low VRAM |

## Training New Models

```bash
# Standard training
python scripts/train.py --profile standard --gpu 1

# Custom settings (stored in training_info.txt)
python scripts/train.py --profile full --r 32 --scheduler cosine
```

Technical details (rank, scheduler, etc.) are saved in `training_info.txt` inside each model folder.

---

*Storage: ~60MB per model*
*Required VRAM: 14GB for inference*