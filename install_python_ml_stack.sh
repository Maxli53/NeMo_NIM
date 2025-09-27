#!/bin/bash
# Install Python and full ML/AI stack

echo "Installing Python ML/AI Stack..."

# Install Python 3.11 and dev tools
sudo apt-get update
sudo apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    cmake \
    git-lfs

# Create virtual environment
cd /media/ubumax/WD_BLACK/AI_Projects/NeMo_GPT
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core ML libraries
pip install \
    transformers \
    datasets \
    tokenizers \
    accelerate \
    bitsandbytes \
    sentencepiece \
    protobuf \
    safetensors \
    huggingface-hub

# Install NeMo Framework
pip install nemo_toolkit[all]

# Install additional AI/ML tools
pip install \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    ipython \
    tqdm \
    wandb \
    tensorboard \
    pytest

# Install LLM-specific tools
pip install \
    langchain \
    llama-index \
    openai \
    anthropic \
    tiktoken \
    vllm

echo ""
echo "✅ Python ML Stack installed in virtual environment!"
echo "⚠️  Activate with: source /media/ubumax/WD_BLACK/AI_Projects/NeMo_GPT/venv/bin/activate"