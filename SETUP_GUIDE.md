# Unsloth GPT-OSS-20B Setup Guide

## Quick Start (5 Minutes)

### Step 1: Create Environment
```bash
cd /media/ubumax/WD_BLACK/AI_Projects/Unsloth_GPT/
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Unsloth
```bash
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
```

### Step 3: Verify Installation
```bash
python -c "from unsloth import FastLanguageModel; print('✅ Unsloth ready!')"
```

That's it! You're ready to train GPT-OSS-20B.

---

## Detailed Setup Guide

### System Requirements Check

#### Hardware Verification
```bash
# Check GPUs (should show 2x RTX 3090)
nvidia-smi --query-gpu=name,memory.total --format=csv

# Expected output:
# NVIDIA GeForce RTX 3090, 24576 MiB
# NVIDIA GeForce RTX 3090, 24576 MiB
```

#### CUDA Verification
```bash
# Check CUDA version (need 12.0+)
nvcc --version

# Check driver version
nvidia-smi | grep "Driver Version"
# Expected: 580.65.06 or higher
```

### Python Environment Setup

#### 1. Create Virtual Environment
```bash
# Navigate to project directory
cd /media/ubumax/WD_BLACK/AI_Projects/Unsloth_GPT/

# Create venv with Python 3.10+
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### 2. Install Unsloth (Official Method)
```bash
# Official installation command from docs.unsloth.ai
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo

# Install additional dependencies without deps conflicts
pip install --no-deps trl peft accelerate bitsandbytes
```

#### 3. Install Supporting Libraries
```bash
# Dataset handling
pip install datasets

# Utilities
pip install transformers huggingface_hub

# Monitoring
pip install tensorboard wandb
```

### GPU Configuration

#### Single GPU Training (Recommended)
```python
# Uses only first RTX 3090 (24GB)
# Perfect for QLoRA - only needs 14GB
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

#### Dual GPU Training (Optional)
```python
# Use both RTX 3090s (48GB total)
# For larger batch sizes or BF16 training
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
```

### Model Download Options

#### Option 1: Automatic (During Training)
```python
# Unsloth will download automatically
model_name = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
```

#### Option 2: Pre-download GGUF
```bash
# Download quantized model for inference
huggingface-cli download unsloth/gpt-oss-20b-GGUF \
    --include "gpt-oss-20b-Q4_K_M.gguf" \
    --local-dir models/gguf/
```

#### Option 3: Download Base Model
```bash
# For custom quantization
huggingface-cli download unsloth/gpt-oss-20b \
    --local-dir models/base/
```

### Verification Script

Create `verify_setup.py`:
```python
#!/usr/bin/env python3
"""Verify Unsloth installation and GPU setup"""

import torch
from unsloth import FastLanguageModel
import sys

def verify_setup():
    print("=" * 50)
    print("Unsloth GPT-OSS-20B Setup Verification")
    print("=" * 50)

    # Check PyTorch and CUDA
    print(f"\n✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ CUDA version: {torch.version.cuda}")

    # Check GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\n✓ Number of GPUs: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")

    # Test Unsloth import
    try:
        from unsloth import FastLanguageModel
        print("\n✓ Unsloth imported successfully")
    except Exception as e:
        print(f"\n✗ Unsloth import failed: {e}")
        sys.exit(1)

    # Check memory for QLoRA
    if num_gpus > 0:
        free_memory = torch.cuda.mem_get_info()[0] / 1e9
        print(f"\n✓ Free VRAM: {free_memory:.1f} GB")
        if free_memory >= 14:
            print("  ✓ Sufficient for QLoRA training (14GB required)")
        else:
            print("  ⚠ May need to close other applications")

    print("\n" + "=" * 50)
    print("✅ Setup verification complete!")
    print("=" * 50)

if __name__ == "__main__":
    verify_setup()
```

Run verification:
```bash
python verify_setup.py
```

### Directory Structure Setup

```bash
# Create project directories
mkdir -p models/{base,gguf,checkpoints,lora_adapters}
mkdir -p data/{raw,processed}
mkdir -p scripts/{training,inference,evaluation}
mkdir -p configs
mkdir -p logs
mkdir -p workspace/tmp

# Create .gitignore
cat > .gitignore << 'EOF'
venv/
*.pyc
__pycache__/
.env
models/*/
data/*/
logs/
workspace/
*.gguf
*.bin
*.safetensors
checkpoints/
wandb/
.ipynb_checkpoints/
EOF
```

### Environment Variables

Create `.env` file:
```bash
# Unsloth settings
export UNSLOTH_USE_FLASH_ATTENTION=1
export TOKENIZERS_PARALLELISM=false

# HuggingFace (optional)
export HF_TOKEN="your_token_here"

# CUDA settings
export CUDA_VISIBLE_DEVICES=0  # or "0,1" for both GPUs

# Paths
export MODEL_PATH="/media/ubumax/WD_BLACK/AI_Projects/Unsloth_GPT/models"
export DATA_PATH="/media/ubumax/WD_BLACK/AI_Projects/Unsloth_GPT/data"
```

Load environment:
```bash
source .env
```

### Common Issues & Solutions

#### Issue 1: CUDA Out of Memory
```python
# Solution: Reduce sequence length
max_seq_length = 8192  # Instead of 16384

# Or enable CPU offloading
load_in_4bit = True
device_map = "auto"
```

#### Issue 2: Slow Installation
```bash
# Use faster mirror
pip install --index-url https://pypi.org/simple/ unsloth
```

#### Issue 3: Import Errors
```bash
# Reinstall with force
pip uninstall unsloth unsloth_zoo -y
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
```

#### Issue 4: GPU Not Detected
```bash
# Check CUDA paths
echo $LD_LIBRARY_PATH
# Should include: /usr/local/cuda/lib64

# Fix if missing
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Quick Test Script

Create `quick_test.py`:
```python
#!/usr/bin/env python3
"""Quick test of model loading"""

from unsloth import FastLanguageModel

print("Loading model (this may take a minute)...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    max_seq_length=1024,  # Small for testing
    dtype=None,
    load_in_4bit=True,
)

print("✅ Model loaded successfully!")
print(f"Model type: {type(model)}")
print(f"Tokenizer vocab size: {len(tokenizer)}")

# Quick inference test
FastLanguageModel.for_inference(model)
prompt = "What is 2+2?"
inputs = tokenizer([prompt], return_tensors="pt").to("cuda:0")
outputs = model.generate(**inputs, max_new_tokens=10)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nTest prompt: {prompt}")
print(f"Model response: {response}")
```

### Next Steps

1. **Run verification**: `python verify_setup.py`
2. **Test model loading**: `python quick_test.py`
3. **Start training**: See `train.py` in scripts/
4. **Configure training**: Edit `configs/training_config.yaml`

### Support

- **Unsloth Discord**: https://discord.gg/unsloth
- **Documentation**: https://docs.unsloth.ai/
- **Issues**: https://github.com/unslothai/unsloth/issues

---

**Setup Time**: ~5-10 minutes
**Download Time**: ~10-30 minutes (depending on model choice)
**Ready to Train**: ✅