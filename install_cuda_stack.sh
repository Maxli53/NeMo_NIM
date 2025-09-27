#!/bin/bash
# Install CUDA Toolkit, cuDNN, and all NVIDIA libraries

echo "Installing NVIDIA CUDA Stack..."

# Remove old CUDA if exists
sudo apt-get remove --purge -y 'cuda*' 'nvidia-cuda-toolkit' 2>/dev/null || true

# Install CUDA 12.6 (latest compatible with driver 580.65)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6

# Install cuDNN 9
sudo apt-get install -y cudnn9-cuda-12

# Install NVIDIA libraries for ML/AI
sudo apt-get install -y \
    libnccl2 \
    libnccl-dev \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    libcublas-12-6 \
    libcublas-dev-12-6 \
    libcufft-12-6 \
    libcufft-dev-12-6 \
    libcurand-12-6 \
    libcurand-dev-12-6 \
    libcusolver-12-6 \
    libcusolver-dev-12-6 \
    libcusparse-12-6 \
    libcusparse-dev-12-6

# Add CUDA to PATH
echo '' >> ~/.bashrc
echo '# CUDA' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

echo ""
echo "✅ CUDA Stack installed!"
echo "⚠️  Run: source ~/.bashrc"
echo "⚠️  Then verify with: nvcc --version"