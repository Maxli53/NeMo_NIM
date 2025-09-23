#!/bin/bash
# Complete ML/AI Stack Installation for WSL
# Run this in WSL Ubuntu 24.04

echo "=============================================="
echo "Installing Complete ML/AI Stack in WSL"
echo "=============================================="

# Activate virtual environment
source ~/cuda_env/bin/activate

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=4

echo ""
echo "Current environment:"
echo "  Python: $(python --version)"
echo "  CUDA: $(nvcc --version | head -1)"
echo ""

# Already installed:
# - torch, torchvision, torchaudio (2.5.1+cu121)
# - flash-attn (2.8.3)
# - triton (3.1.0)
# - einops (0.8.1)

echo "1. Installing Hugging Face Stack..."
pip install transformers datasets tokenizers evaluate
pip install accelerate optimum peft trl
pip install sentence-transformers diffusers

echo ""
echo "2. Installing LLM Optimization Libraries..."
pip install bitsandbytes  # 8-bit/4-bit quantization
pip install auto-gptq[triton]  # GPTQ with Triton
pip install awq  # Activation-aware Weight Quantization
pip install llama-cpp-python  # CPU/GPU inference

echo ""
echo "3. Installing Memory Efficient Attention..."
pip install xformers  # Should work with Python 3.12
pip install rotary-embedding-torch  # RoPE
pip install local-attention  # Local attention patterns

echo ""
echo "4. Installing DeepSpeed & Distributed Training..."
CUDA_HOME=/usr/local/cuda-12.0 pip install deepspeed
pip install fairscale
pip install colossalai  # Large-scale parallel training

echo ""
echo "5. Installing Inference Optimization..."
pip install vllm  # PagedAttention, continuous batching
pip install tensorrt-llm  # If available
pip install onnx onnxruntime-gpu
pip install ctransformers  # Fast inference

echo ""
echo "6. Installing CUDA-specific Tools..."
pip install cupy-cuda12x  # NumPy-like API for CUDA
pip install pycuda  # Python wrapper for CUDA
pip install cuda-python  # Official NVIDIA bindings

echo ""
echo "7. Installing Vector Databases & RAG..."
pip install faiss-cpu faiss-gpu
pip install chromadb
pip install langchain langchain-community
pip install llama-index

echo ""
echo "8. Installing Monitoring & Profiling..."
pip install wandb tensorboard
pip install nvitop gpustat py3nvml
pip install memory-profiler line-profiler

echo ""
echo "9. Installing API & Serving..."
pip install fastapi uvicorn[standard]
pip install gradio streamlit
pip install ray[serve] bentoml

echo ""
echo "10. Installing JAX (Alternative to PyTorch)..."
pip install jax[cuda12]
pip install flax optax

echo ""
echo "11. Installing Additional Tools..."
pip install jupyterlab ipywidgets
pip install datasets polars pyarrow
pip install tiktoken sentencepiece

echo ""
echo "=============================================="
echo "Installation Complete!"
echo "=============================================="
echo ""
echo "Installed packages summary:"
pip list | grep -E "torch|transformers|accelerate|flash|triton|deepspeed|xformers|vllm|jax"
echo ""
echo "To use this environment:"
echo "  source ~/cuda_env/bin/activate"
echo "  python your_script.py"
echo ""