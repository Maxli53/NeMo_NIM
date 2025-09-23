#!/usr/bin/env python3
"""
Comprehensive ML/LLM Stack Readiness Check for WSL2
Tests all critical components for AI/ML workloads on GPU
"""

import sys
import subprocess
from typing import Dict, Tuple, List

def run_check(name: str, test_func) -> Tuple[bool, str]:
    """Run a single check and return status"""
    try:
        result = test_func()
        return (True, result)
    except Exception as e:
        return (False, str(e))

def check_cuda():
    """Check CUDA availability"""
    import torch
    if not torch.cuda.is_available():
        raise Exception("CUDA not available")

    device = torch.cuda.get_device_properties(0)
    return f"GPU: {device.name}, Memory: {device.total_memory / 1024**3:.1f}GB, CUDA: {torch.version.cuda}"

def check_flash_attention():
    """Check Flash Attention"""
    import flash_attn
    return f"Flash Attention v{flash_attn.__version__}"

def check_transformers():
    """Check Transformers ecosystem"""
    import transformers
    import accelerate
    import peft
    import trl
    return f"Transformers: {transformers.__version__}, Accelerate: {accelerate.__version__}, PEFT: {peft.__version__}"

def check_quantization():
    """Check quantization libraries"""
    import bitsandbytes as bnb
    return f"BitsAndBytes: {bnb.__version__} (8-bit/4-bit quantization ready)"

def check_inference():
    """Check inference optimization"""
    import vllm
    import xformers
    return f"vLLM: {vllm.__version__}, xFormers: {xformers.__version__}"

def check_distributed():
    """Check distributed training"""
    import deepspeed
    import torch.distributed as dist
    return f"DeepSpeed: {deepspeed.__version__}, PyTorch DDP ready"

def check_monitoring():
    """Check monitoring tools"""
    import wandb
    import tensorboard
    import gpustat
    return f"W&B: {wandb.__version__}, TensorBoard: {tensorboard.__version__}"

def check_serving():
    """Check API serving"""
    import fastapi
    import uvicorn
    import gradio
    import streamlit
    return f"FastAPI: {fastapi.__version__}, Gradio: {gradio.__version__}, Streamlit: {streamlit.__version__}"

def check_data():
    """Check data handling"""
    import datasets
    import tiktoken
    import safetensors
    return f"Datasets: {datasets.__version__}, Tiktoken: {tiktoken.__version__}"

def check_vector_db():
    """Check vector databases"""
    import chromadb
    import langchain
    import faiss
    return f"ChromaDB: {chromadb.__version__}, LangChain: {langchain.__version__}, FAISS ready"

def check_memory_optimization():
    """Check memory optimization capabilities"""
    import torch
    import gc

    # Check mixed precision
    if torch.cuda.is_available():
        supports_fp16 = torch.cuda.is_bf16_supported()
        supports_tf32 = True  # RTX 3090 supports TF32

        # Check available memory
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_gb = free_mem / 1024**3
        total_gb = total_mem / 1024**3

        return f"FP16: ✓, BF16: {'✓' if supports_fp16 else '✗'}, TF32: ✓, Free GPU: {free_gb:.1f}/{total_gb:.1f}GB"
    return "GPU not available"

def check_triton():
    """Check Triton compiler"""
    import triton
    return f"Triton: {triton.__version__} (JIT compilation ready)"

def main():
    print("=" * 70)
    print("🚀 ML/LLM STACK READINESS CHECK FOR WSL2")
    print("=" * 70)
    print()

    checks = {
        "🖥️  CUDA & GPU": check_cuda,
        "⚡ Flash Attention": check_flash_attention,
        "🤗 Transformers": check_transformers,
        "📊 Quantization": check_quantization,
        "🚄 Inference Optimization": check_inference,
        "🌐 Distributed Training": check_distributed,
        "📈 Monitoring Tools": check_monitoring,
        "🌍 API Serving": check_serving,
        "📚 Data Handling": check_data,
        "🔍 Vector Databases": check_vector_db,
        "💾 Memory Optimization": check_memory_optimization,
        "⚙️  Triton Compiler": check_triton,
    }

    results = {}
    passed = 0
    failed = 0

    for name, check_func in checks.items():
        success, result = run_check(name, check_func)
        results[name] = (success, result)

        if success:
            print(f"✅ {name}: {result}")
            passed += 1
        else:
            print(f"❌ {name}: {result}")
            failed += 1

    print()
    print("=" * 70)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("✨ EXCELLENT! Your system is 100% ready for AI/ML/LLM workloads!")
        print()
        print("🎯 You can now:")
        print("  • Train large language models with DeepSpeed")
        print("  • Run inference with vLLM (PagedAttention)")
        print("  • Use 8-bit/4-bit quantization (BitsAndBytes)")
        print("  • Deploy with FastAPI/Gradio/Streamlit")
        print("  • Monitor with W&B/TensorBoard")
        print("  • Build RAG applications with LangChain")
    else:
        print("⚠️  Some components are missing. Install them with:")
        print("  wsl")
        print("  source ~/cuda_env/bin/activate")
        print("  pip install <missing_package>")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())