#!/bin/bash

# NeMo GPT-OSS Project Setup Script
# This script sets up the complete environment for NeMo and NIM

set -e

echo "========================================"
echo "NeMo GPT-OSS Project Setup"
echo "========================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }

# Check if running in WSL2
if grep -qi microsoft /proc/version; then
    print_status "Running in WSL2"
else
    print_warning "Not in WSL2. Some features may not work correctly."
fi

# Step 1: Check GPU
echo -e "\n1. Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    print_status "GPU detected"
else
    print_error "GPU not detected. Please check your CUDA installation."
    exit 1
fi

# Step 2: Install NVIDIA Container Toolkit
echo -e "\n2. Installing NVIDIA Container Toolkit..."
if ! command -v nvidia-container-cli &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    print_status "NVIDIA Container Toolkit installed"
else
    print_status "NVIDIA Container Toolkit already installed"
fi

# Step 3: Pull NeMo Container
echo -e "\n3. Pulling NeMo GPT-OSS container..."
docker pull nvcr.io/nvidia/nemo:25.07.gpt_oss || {
    print_warning "Failed to pull container. You may need to login to NGC."
    echo "Visit https://ngc.nvidia.com to get credentials"
    echo "Run: docker login nvcr.io"
}

# Step 4: Build custom container
echo -e "\n4. Building custom NeMo container..."
cd nim/docker
docker-compose build nemo-dev
cd ../..
print_status "Custom container built"

# Step 5: Create required directories
echo -e "\n5. Setting up workspace directories..."
mkdir -p workspace/{experiments,checkpoints,data,outputs}
mkdir -p workspace/checkpoints/{base,converted,quantized}
mkdir -p workspace/outputs/{logs,tensorboard,evaluations}
print_status "Workspace directories created"

# Step 6: Install Python dependencies (for local development)
echo -e "\n6. Installing Python dependencies..."
if [ -f "requirements/base.txt" ]; then
    pip install -r requirements/base.txt
    print_status "Python dependencies installed"
else
    print_warning "requirements/base.txt not found"
fi

# Step 7: Verify installation
echo -e "\n7. Verifying installation..."
docker run --rm --gpus all nvcr.io/nvidia/nemo:25.07.gpt_oss python -c "
import torch
import nemo
print(f'PyTorch version: {torch.__version__}')
print(f'NeMo version: {nemo.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "========================================"
print_status "Setup completed successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Place your GPT-OSS-20B model in: workspace/checkpoints/base/"
echo "2. Import model: docker-compose run nemo-dev python /scripts/import_gpt_oss.py --source /workspace/checkpoints/base/gpt-oss-20b"
echo "3. Start training: docker-compose up nemo-train"
echo "4. Run inference: docker-compose up nemo-inference"
echo ""
echo "For development:"
echo "  docker-compose run --rm nemo-dev bash"
echo ""
echo "Documentation:"
echo "  - NeMo: https://docs.nvidia.com/nemo-framework/"
echo "  - GPT-OSS: https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/gpt_oss.html"