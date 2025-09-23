#!/bin/bash
# Run Python scripts with Flash Attention enabled in WSL

echo "======================================"
echo "Starting WSL CUDA Environment"
echo "======================================"
echo ""
echo "Environment:"
echo "  - CUDA 12.0 with nvcc"
echo "  - PyTorch 2.5.1+cu121"
echo "  - Flash Attention 2.8.3"
echo "  - RTX 3090 (24GB)"
echo ""

# Activate the virtual environment
source ~/cuda_env/bin/activate

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Check if a Python script was provided
if [ $# -eq 0 ]; then
    echo "Usage: ./run_with_flash.sh <python_script.py>"
    echo ""
    echo "Starting Python interactive shell with Flash Attention..."
    python3
else
    echo "Running: $1"
    echo "======================================"
    python3 "$@"
fi