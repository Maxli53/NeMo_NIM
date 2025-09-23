#!/bin/bash
# System-level dependencies for ML/AI in WSL
# Run with: sudo bash install_system_deps_wsl.sh

echo "=============================================="
echo "Installing System Dependencies for ML/AI"
echo "=============================================="

# Update package list
echo "Updating package lists..."
apt-get update

# 1. Essential build tools (if not already installed)
echo ""
echo "1. Installing build tools..."
apt-get install -y build-essential cmake ninja-build
apt-get install -y python3-dev python3-pip python3-venv
apt-get install -y git wget curl htop tmux vim

# 2. Linear algebra libraries
echo ""
echo "2. Installing math libraries..."
apt-get install -y libopenblas-dev liblapack-dev
apt-get install -y libatlas-base-dev gfortran
apt-get install -y libblas-dev liblapacke-dev

# 3. HDF5 for large datasets
echo ""
echo "3. Installing HDF5..."
apt-get install -y libhdf5-dev hdf5-tools

# 4. Image processing
echo ""
echo "4. Installing image libraries..."
apt-get install -y libjpeg-dev libpng-dev libtiff-dev
apt-get install -y libavcodec-dev libavformat-dev libswscale-dev

# 5. OpenMPI for distributed training
echo ""
echo "5. Installing OpenMPI..."
apt-get install -y openmpi-bin openmpi-common libopenmpi-dev

# 6. NCCL for multi-GPU (if not from CUDA)
echo ""
echo "6. Checking NCCL..."
if ! dpkg -l | grep -q libnccl; then
    echo "Installing NCCL..."
    apt-get install -y libnccl2 libnccl-dev
else
    echo "NCCL already installed"
fi

# 7. Additional CUDA tools
echo ""
echo "7. Installing additional CUDA tools..."
apt-get install -y nvidia-cuda-toolkit-gcc  # CUDA-aware GCC
apt-get install -y nvidia-visual-profiler  # Visual profiler

# 8. Python optimization tools
echo ""
echo "8. Installing Python tools..."
apt-get install -y python3-numpy python3-scipy python3-pandas
apt-get install -y cython3

# 9. Compression tools
echo ""
echo "9. Installing compression tools..."
apt-get install -y zlib1g-dev liblzma-dev libbz2-dev

# 10. SSL and cryptography
echo ""
echo "10. Installing SSL libraries..."
apt-get install -y libssl-dev libffi-dev

# 11. Database drivers (for vector DBs)
echo ""
echo "11. Installing database libraries..."
apt-get install -y libpq-dev  # PostgreSQL
apt-get install -y libsqlite3-dev  # SQLite

# 12. Monitoring tools
echo ""
echo "12. Installing monitoring tools..."
apt-get install -y nvtop  # NVIDIA GPU monitor
apt-get install -y iotop sysstat

# 13. Check for cuDNN
echo ""
echo "13. Checking cuDNN..."
if ! dpkg -l | grep -q libcudnn; then
    echo "cuDNN not installed. To install:"
    echo "  1. Download from: https://developer.nvidia.com/cudnn"
    echo "  2. sudo dpkg -i cudnn-local-repo-*.deb"
    echo "  3. sudo apt-get update"
    echo "  4. sudo apt-get install libcudnn9-cuda-12"
else
    echo "cuDNN already installed"
fi

# 14. Check for TensorRT
echo ""
echo "14. Checking TensorRT..."
if ! dpkg -l | grep -q tensorrt; then
    echo "TensorRT not installed. To install:"
    echo "  1. Download from: https://developer.nvidia.com/tensorrt"
    echo "  2. sudo dpkg -i nv-tensorrt-local-repo-*.deb"
    echo "  3. sudo apt-get update"
    echo "  4. sudo apt-get install tensorrt"
else
    echo "TensorRT already installed"
fi

echo ""
echo "=============================================="
echo "System Dependencies Installation Complete!"
echo "=============================================="
echo ""
echo "Installed packages:"
echo "  - Build tools: gcc, g++, cmake, ninja"
echo "  - Math libraries: OpenBLAS, LAPACK, ATLAS"
echo "  - MPI: OpenMPI for distributed training"
echo "  - CUDA: Full toolkit with nvcc"
echo "  - Monitoring: nvtop, htop, gpustat"
echo ""
echo "Next steps:"
echo "  1. Run: ./install_ml_stack_wsl.sh"
echo "  2. Install cuDNN and TensorRT if needed"
echo ""