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

## Model Setup & Cache Management

### Model Download
The GPT-OSS-20B 4-bit model (~12GB) needs to be downloaded correctly for Unsloth to find it:

#### Option 1: Let Unsloth Handle It (Recommended)
Simply use the model in your code and Unsloth will download to the correct location:
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)
```

#### Option 2: Manual Download to HF Cache
If you've already downloaded the model elsewhere, you need to place it in the HF cache:

1. **Expected Cache Structure**:
```
~/.cache/huggingface/hub/models--unsloth--gpt-oss-20b-unsloth-bnb-4bit/
├── blobs/
│   ├── f229f4...8b89 (model-00001, 3.8GB)
│   ├── 2b75ef...0a30 (model-00002, 3.8GB)
│   ├── 76cc7c...f90  (model-00003, 3.2GB)
│   └── 31709e...e7bc (model-00004, 1.1GB)
└── snapshots/093fba.../
    ├── model-00001-of-00004.safetensors -> ../../blobs/f229f4...
    └── [symlinks to other shards]
```

2. **Fix Existing Download**:
```bash
# If you downloaded to custom directory, copy to cache:
SOURCE_DIR="/path/to/your/downloaded/model"
CACHE_DIR="$HOME/.cache/huggingface/hub/models--unsloth--gpt-oss-20b-unsloth-bnb-4bit"

# Create structure
mkdir -p $CACHE_DIR/blobs
mkdir -p $CACHE_DIR/snapshots/093fba6992ef5a7152481afec0bdfca1ac486998

# Copy with correct hash names (required!)
cp $SOURCE_DIR/model-00001-of-00004.safetensors $CACHE_DIR/blobs/f229f4364b1f2cfa7df0ced4d22777145d3d552f95512f98b20493ea094e8b89
cp $SOURCE_DIR/model-00002-of-00004.safetensors $CACHE_DIR/blobs/2b75ef14502b54c47cad18bb6c2f96e919991ba994a3331799e38891ac290a30
cp $SOURCE_DIR/model-00003-of-00004.safetensors $CACHE_DIR/blobs/76cc7c3bf1cd287a8a7cea77e00eb45ce6d6a0fbc7084d29a347eed0b398af90
cp $SOURCE_DIR/model-00004-of-00004.safetensors $CACHE_DIR/blobs/31709e4bd1403df4437091d952e2ec837efbbfc06337b4b2b6c6875491e0e7bc

# Create symlinks
cd $CACHE_DIR/snapshots/093fba6992ef5a7152481afec0bdfca1ac486998
ln -sf ../../blobs/f229f4364b1f2cfa7df0ced4d22777145d3d552f95512f98b20493ea094e8b89 model-00001-of-00004.safetensors
ln -sf ../../blobs/2b75ef14502b54c47cad18bb6c2f96e919991ba994a3331799e38891ac290a30 model-00002-of-00004.safetensors
ln -sf ../../blobs/76cc7c3bf1cd287a8a7cea77e00eb45ce6d6a0fbc7084d29a347eed0b398af90 model-00003-of-00004.safetensors
ln -sf ../../blobs/31709e4bd1403df4437091d952e2ec837efbbfc06337b4b2b6c6875491e0e7bc model-00004-of-00004.safetensors
```

### Important: Use Model NAME, Not Path!
❌ **WRONG** - Will cause re-download:
```python
model_name="./models/gpt-oss-20b-4bit"  # Local path doesn't work!
```

✅ **CORRECT** - Uses cached model:
```python
model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit"  # Model name from HF
```

### Common Issues
- **"Fetching 4 files" despite having model**: Model not in correct cache location
- **5+ minute load times**: Model being re-downloaded, check cache structure
- **"GptOssForCausalLM not found"**: Use Unsloth's FastLanguageModel, not raw transformers

## Training

### Profiles
- `quick_test`: 30 steps validation
- `standard`: 100 steps training
- `high_quality`: r=32, 2 epochs
- `memory_efficient`: r=8, 1024 context

```bash
# Standard training
python scripts/train_advanced.py --profile standard --validate

# Resume from checkpoint
python scripts/train_advanced.py --profile standard \
    --resume_from_checkpoint ./outputs/checkpoint-100

# Custom dataset
python scripts/prepare_dataset.py \
    --input_file my_data.json \
    --format alpaca \
    --output_dir ./data/processed

python scripts/train_simple.py \
    --dataset_name ./data/processed/hf_dataset/train \
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

## Evaluation

```bash
# Evaluate model quality
python scripts/evaluate.py \
    --model_path ./final_model \
    --dataset HuggingFaceH4/Multilingual-Thinking \
    --compare_base

# Benchmark performance
python scripts/benchmark.py --model_path ./final_model
```

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
│   ├── train_advanced.py         # Advanced with monitoring & resume
│   ├── inference.py              # Generation
│   ├── export_to_llama.py       # GGUF export
│   ├── benchmark.py              # Performance test
│   ├── prepare_dataset.py       # Dataset preparation
│   └── evaluate.py               # Model evaluation
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

## Advanced: GRPO/RL Training (NEW!)

```bash
# Train with reinforcement learning (15GB VRAM)
python scripts/train_grpo.py \
    --task code_optimization \
    --max_steps 100 \
    --num_generations 2

# Available tasks: code_optimization, reasoning, creative
```

See `docs/GRPO_RL_GUIDE.md` for complete RL documentation.

## Resources

- [Unsloth Docs](https://docs.unsloth.ai/)
- [GPT-OSS Guide](https://docs.unsloth.ai/new/gpt-oss-how-to-run-and-fine-tune)
- [GRPO Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb)
- [Model Hub](https://huggingface.co/unsloth/gpt-oss-20b-GGUF)

---
Hardware: 2x RTX 3090 (24GB each) | Created: 2025-09-29