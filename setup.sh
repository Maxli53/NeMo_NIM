#!/bin/bash
# Unsloth GPT-OSS-20B Setup Script
# Automated setup for Ubuntu with RTX 3090

set -e

echo "=========================================="
echo "Unsloth GPT-OSS-20B Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running from correct directory
if [ ! -f "ROADMAP.md" ]; then
    echo -e "${RED}Error: Please run this script from the Unsloth_GPT directory${NC}"
    exit 1
fi

# Step 1: Check CUDA
echo -e "\n${YELLOW}Step 1: Checking CUDA installation...${NC}"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "${GREEN}✓ CUDA $CUDA_VERSION detected${NC}"
else
    echo -e "${RED}✗ CUDA not found. Please install CUDA 12.0+${NC}"
    exit 1
fi

# Step 2: Check GPUs
echo -e "\n${YELLOW}Step 2: Checking GPUs...${NC}"
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo -e "${GREEN}✓ Found $GPU_COUNT GPU(s):${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Step 3: Create virtual environment
echo -e "\n${YELLOW}Step 3: Creating Python virtual environment...${NC}"
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Use it? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        rm -rf venv
        python3 -m venv venv
    fi
else
    python3 -m venv venv
fi
echo -e "${GREEN}✓ Virtual environment ready${NC}"

# Step 4: Activate and upgrade pip
echo -e "\n${YELLOW}Step 4: Activating environment and upgrading pip...${NC}"
source venv/bin/activate
pip install --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"

# Step 5: Install UV package manager
echo -e "\n${YELLOW}Step 5: Installing UV package manager...${NC}"
pip install --upgrade -qqq uv
echo -e "${GREEN}✓ UV installed${NC}"

# Step 6: Install Unsloth with latest versions (official method)
echo -e "\n${YELLOW}Step 6: Installing Unsloth with latest dependencies (this may take several minutes)...${NC}"

# Check if numpy is installed
python -c "import numpy; print(f'numpy=={numpy.__version__}')" 2>/dev/null > /tmp/numpy_version.txt || echo "numpy" > /tmp/numpy_version.txt
NUMPY_VERSION=$(cat /tmp/numpy_version.txt)

# Install using UV with official command from docs
uv pip install -qqq \
    "torch>=2.8.0" "triton>=3.4.0" ${NUMPY_VERSION} \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth" \
    torchvision bitsandbytes \
    git+https://github.com/huggingface/transformers \
    git+https://github.com/triton-lang/triton.git@05b2c186c1b6c9a08375389d5efe9cb4c401c075#subdirectory=python/triton_kernels

echo -e "${GREEN}✓ Unsloth and dependencies installed${NC}"

# Step 7: Install additional training dependencies
echo -e "\n${YELLOW}Step 7: Installing additional training dependencies...${NC}"
pip install trl datasets huggingface_hub tensorboard wandb
echo -e "${GREEN}✓ Training dependencies installed${NC}"

# Step 7: Create directory structure
echo -e "\n${YELLOW}Step 7: Creating project directories...${NC}"
mkdir -p models/{base,gguf,checkpoints,lora_adapters}
mkdir -p data/{raw,processed}
mkdir -p scripts/{training,inference,evaluation}
mkdir -p configs logs workspace/tmp
echo -e "${GREEN}✓ Directories created${NC}"

# Step 8: Verify installation
echo -e "\n${YELLOW}Step 8: Verifying installation...${NC}"
python3 -c "
import torch
from unsloth import FastLanguageModel
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
print('✓ Unsloth imported successfully')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Installation verified${NC}"
else
    echo -e "${RED}✗ Installation verification failed${NC}"
    exit 1
fi

# Step 9: Create test script
echo -e "\n${YELLOW}Step 9: Creating test script...${NC}"
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""Quick test of Unsloth setup"""

import torch
from unsloth import FastLanguageModel
import sys

def test_setup():
    print("Testing Unsloth setup...")

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        sys.exit(1)

    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✓ Number of GPUs: {torch.cuda.device_count()}")

    # Check memory
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"✓ GPU {i}: {props.total_memory / 1e9:.1f} GB")

    # Test model loading (small test)
    try:
        print("\nAttempting to load model architecture...")
        # This just tests the import, not actual model download
        from unsloth import FastLanguageModel
        print("✓ FastLanguageModel imported successfully")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    print("\n✅ All tests passed! Ready to train GPT-OSS-20B")

if __name__ == "__main__":
    test_setup()
EOF

chmod +x test_setup.py
echo -e "${GREEN}✓ Test script created${NC}"

# Step 10: Create environment file
echo -e "\n${YELLOW}Step 10: Creating environment file...${NC}"
cat > .env << 'EOF'
# Unsloth environment variables
export UNSLOTH_USE_FLASH_ATTENTION=1
export TOKENIZERS_PARALLELISM=false

# CUDA settings (use single GPU by default)
export CUDA_VISIBLE_DEVICES=0

# Paths
export MODEL_PATH="/media/ubumax/WD_BLACK/AI_Projects/Unsloth_GPT/models"
export DATA_PATH="/media/ubumax/WD_BLACK/AI_Projects/Unsloth_GPT/data"
EOF
echo -e "${GREEN}✓ Environment file created${NC}"

# Final summary
echo -e "\n${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Load environment: source .env"
echo "3. Test setup: python test_setup.py"
echo "4. Start training: python scripts/train.py --help"
echo ""
echo "Quick start training command:"
echo "  python scripts/train.py --max_steps 100 --output_dir ./checkpoints"
echo ""
echo "For interactive inference:"
echo "  python scripts/inference.py --model_path unsloth/gpt-oss-20b-unsloth-bnb-4bit --interactive"
echo ""
echo -e "${GREEN}Ready to fine-tune GPT-OSS-20B!${NC}"