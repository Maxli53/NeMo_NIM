#!/bin/bash
# Install CUDA Toolkit in WSL for Flash Attention

echo "==============================================="
echo "Installing CUDA Toolkit 12.1 in WSL"
echo "==============================================="

# Update package list
echo "1. Updating package list..."
sudo apt-get update

# Install required dependencies
echo "2. Installing dependencies..."
sudo apt-get install -y wget build-essential

# Download and install CUDA toolkit for WSL
echo "3. Setting up CUDA repository..."
wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

# Update package list again
sudo apt-get update

# Install CUDA toolkit (matching PyTorch's CUDA 12.1)
echo "4. Installing CUDA toolkit 12.1..."
sudo apt-get -y install cuda-toolkit-12-1

# Set environment variables
echo "5. Setting environment variables..."
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Apply changes
source ~/.bashrc

# Verify installation
echo "6. Verifying installation..."
if [ -f /usr/local/cuda-12.1/bin/nvcc ]; then
    echo "CUDA toolkit installed successfully!"
    /usr/local/cuda-12.1/bin/nvcc --version
else
    echo "Warning: nvcc not found at expected location"
fi

echo ""
echo "Setup complete! Now you can install flash-attn:"
echo "export CUDA_HOME=/usr/local/cuda-12.1"
echo "pip install flash-attn --no-build-isolation"