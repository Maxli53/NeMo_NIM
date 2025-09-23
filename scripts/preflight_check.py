#!/usr/bin/env python3
"""
Pre-flight Check for MoE Testing Environment
Verifies CUDA, PyTorch, GPU, memory, and all optimization libraries
"""

import torch
import importlib.util
import json
from datetime import datetime
import os
import sys

def check_cuda():
    """Check CUDA availability and configuration"""
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    devices = [torch.cuda.get_device_name(i) for i in range(device_count)] if cuda_available else []

    result = {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "devices": devices,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if cuda_available else None,
    }

    if cuda_available:
        result.update({
            "current_device": torch.cuda.current_device(),
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "total_memory_gb": [torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(device_count)],
            "capability": [f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
                          for i in range(device_count)]
        })

    return result

def check_library(lib_name):
    """Check if a library is installed and get version if possible"""
    spec = importlib.util.find_spec(lib_name)
    if spec is None:
        return {"installed": False, "version": None}

    try:
        module = __import__(lib_name)
        version = getattr(module, "__version__", "unknown")
        return {"installed": True, "version": version}
    except:
        return {"installed": True, "version": "unknown"}

def check_memory():
    """Check current GPU memory status"""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "reserved_gb": 0, "free_gb": 0}

    allocated_gb = torch.cuda.memory_allocated() / 1e9
    reserved_gb = torch.cuda.memory_reserved() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_gb = total_gb - reserved_gb

    return {
        "allocated_gb": allocated_gb,
        "reserved_gb": reserved_gb,
        "free_gb": free_gb,
        "total_gb": total_gb,
        "usage_percent": (reserved_gb / total_gb) * 100
    }

def check_flash_attention():
    """Check Flash Attention availability"""
    result = {"sdpa_available": False, "flash_attn_available": False, "xformers_available": False}

    # Check PyTorch SDPA
    try:
        from torch.nn.functional import scaled_dot_product_attention
        result["sdpa_available"] = True

        # Check which backends are available
        if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
            result["flash_backend"] = torch.backends.cuda.flash_sdp_enabled()
        else:
            result["flash_backend"] = False

    except ImportError:
        pass

    # Check Flash Attention package
    try:
        import flash_attn
        result["flash_attn_available"] = True
        result["flash_attn_version"] = flash_attn.__version__
    except ImportError:
        pass

    # Check xFormers
    try:
        import xformers
        import xformers.ops
        result["xformers_available"] = True
        result["xformers_version"] = xformers.__version__
    except ImportError:
        pass

    return result

def preflight_report():
    """Generate comprehensive pre-flight report"""
    report = {}
    report["timestamp"] = datetime.now().isoformat()

    # CUDA and GPU
    print("Checking CUDA and GPU...")
    report["cuda"] = check_cuda()

    # Memory
    print("Checking memory...")
    report["memory"] = check_memory()

    # Optimization Libraries
    print("Checking optimization libraries...")
    libs_to_check = {
        "bitsandbytes": "INT8 quantization",
        "safetensors": "Efficient model loading",
        "transformers": "Model architecture",
        "accelerate": "Training optimization",
        "xformers": "Memory-efficient attention",
        "triton": "GPU kernels",
        "vllm": "Production inference",
        "flash_attn": "Flash Attention v2"
    }

    report["libraries"] = {}
    for lib, description in libs_to_check.items():
        lib_info = check_library(lib)
        lib_info["description"] = description
        report["libraries"][lib] = lib_info

    # Flash Attention specific checks
    print("Checking Flash Attention backends...")
    report["flash_attention"] = check_flash_attention()

    # Environment Variables
    print("Checking environment variables...")
    env_vars = ["CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH",
                "TORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS"]
    report["env_vars"] = {var: os.getenv(var) for var in env_vars}

    # Feature Flags
    FEATURE_FLAGS = {
        "fp16_baseline": True,
        "int8_quantization": True,
        "mixed_precision": True,
        "sdpa_attention": True
    }
    report["feature_flags"] = FEATURE_FLAGS

    # Model path check
    model_path = "gpt-oss-20b/original/model.safetensors"
    report["model_exists"] = os.path.exists(model_path)
    if report["model_exists"]:
        report["model_size_gb"] = os.path.getsize(model_path) / 1e9

    # Save report
    with open("tests/test_results/preflight_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("PRE-FLIGHT CHECK SUMMARY")
    print("="*60)

    # CUDA Status
    cuda_status = report["cuda"]
    if cuda_status["cuda_available"]:
        print(f"✅ CUDA: Available (v{cuda_status['cuda_version']})")
        print(f"✅ PyTorch: {cuda_status['pytorch_version']}")
        print(f"✅ GPU: {', '.join(cuda_status['devices'])}")
        print(f"✅ VRAM: {cuda_status['total_memory_gb'][0]:.1f} GB")
        print(f"✅ Compute Capability: {cuda_status['capability'][0]}")
    else:
        print("❌ CUDA: Not available")

    # Memory Status
    mem = report["memory"]
    print(f"\nMemory Status:")
    print(f"  Allocated: {mem['allocated_gb']:.2f} GB")
    print(f"  Reserved: {mem['reserved_gb']:.2f} GB")
    print(f"  Free: {mem['free_gb']:.2f} GB")
    print(f"  Usage: {mem['usage_percent']:.1f}%")

    # Libraries
    print(f"\nOptimization Libraries:")
    for lib, info in report["libraries"].items():
        status = "✅" if info["installed"] else "❌"
        version = f"v{info['version']}" if info["version"] and info["version"] != "unknown" else ""
        desc = info["description"]
        print(f"  {status} {lib:15} {version:12} - {desc}")

    # Flash Attention
    flash = report["flash_attention"]
    print(f"\nFlash Attention Support:")
    print(f"  {'✅' if flash['sdpa_available'] else '❌'} PyTorch SDPA: {'Available' if flash['sdpa_available'] else 'Not available'}")
    print(f"  {'✅' if flash.get('flash_backend', False) else '❌'} Flash Backend: {'Enabled' if flash.get('flash_backend', False) else 'Disabled'}")
    print(f"  {'✅' if flash['xformers_available'] else '❌'} xFormers: {'v' + flash.get('xformers_version', '') if flash['xformers_available'] else 'Not installed'}")

    # Model
    print(f"\nModel Check:")
    if report["model_exists"]:
        print(f"  ✅ Model found: {report['model_size_gb']:.1f} GB")
    else:
        print(f"  ❌ Model not found at: {model_path}")

    # Environment
    print(f"\nEnvironment Variables:")
    for var, value in report["env_vars"].items():
        status = "✅" if value else "⚠️"
        print(f"  {status} {var}: {value if value else 'Not set'}")

    # Overall readiness
    print("\n" + "="*60)
    critical_libs = ["bitsandbytes", "safetensors", "transformers"]
    all_critical = all(report["libraries"][lib]["installed"] for lib in critical_libs)

    if cuda_status["cuda_available"] and all_critical and report["model_exists"]:
        print("✅ READY FOR TESTING")
        print("Run: wsl bash -c 'source ~/cuda_env/bin/activate && python tests/test_real_moe_performance.py'")
    else:
        print("❌ NOT READY - Fix issues above before testing")

    print("="*60)
    print(f"\nFull report saved to: tests/test_results/preflight_report.json")

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("tests/test_results", exist_ok=True)
    preflight_report()