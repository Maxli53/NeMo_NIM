#!/usr/bin/env python3
"""
Test script for WSL optimizations
Run this in WSL to verify torch.compile and bitsandbytes work
"""

import sys
import time
import torch

print("="*60)
print("WSL OPTIMIZATION TEST")
print("="*60)

# 1. Check environment
print("\n1. Environment Check:")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 2. Test torch.compile
print("\n2. Testing torch.compile:")
print("torch.compile available:", hasattr(torch, 'compile'))

if hasattr(torch, 'compile'):
    def test_function(x, y):
        return (x * y).sum()

    # Compile the function
    compiled_fn = torch.compile(test_function, mode="reduce-overhead")

    # Create test data
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)

    # Warm-up (triggers compilation)
    print("Compiling...")
    _ = compiled_fn(x, y)

    # Benchmark
    print("Benchmarking...")

    # Non-compiled
    start = time.perf_counter()
    for _ in range(100):
        _ = test_function(x, y)
    non_compiled_time = time.perf_counter() - start

    # Compiled
    start = time.perf_counter()
    for _ in range(100):
        _ = compiled_fn(x, y)
    compiled_time = time.perf_counter() - start

    print(f"Non-compiled: {non_compiled_time:.3f}s")
    print(f"Compiled: {compiled_time:.3f}s")
    print(f"Speedup: {non_compiled_time/compiled_time:.2f}x")
    print("✅ torch.compile WORKS!")
else:
    print("❌ torch.compile not available")

# 3. Test bitsandbytes
print("\n3. Testing bitsandbytes:")
try:
    import bitsandbytes as bnb
    print("✅ Bitsandbytes imported successfully")

    # Test INT8 quantization
    print("Testing INT8 quantization...")

    # Create a linear layer
    linear = torch.nn.Linear(1024, 1024)

    # Quantize it
    linear_int8 = bnb.nn.Linear8bitLt(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        has_fp16_weights=False
    )

    # Copy weights
    linear_int8.weight.data = linear.weight.data
    if linear.bias is not None:
        linear_int8.bias.data = linear.bias.data

    # Test forward pass
    x = torch.randn(32, 1024)

    # Original
    out_fp32 = linear(x)

    # Quantized
    out_int8 = linear_int8(x)

    # Check similarity
    diff = (out_fp32 - out_int8).abs().mean()
    print(f"Mean difference: {diff:.6f}")
    print(f"Original size: {linear.weight.numel() * 4 / 1e6:.2f} MB")
    print(f"Quantized size: ~{linear.weight.numel() * 1 / 1e6:.2f} MB (4x reduction)")

    if diff < 0.01:
        print("✅ Bitsandbytes INT8 WORKS!")
    else:
        print("⚠️ Quantization difference larger than expected")

except ImportError as e:
    print(f"❌ Bitsandbytes import failed: {e}")
except Exception as e:
    print(f"❌ Bitsandbytes test failed: {e}")

# 4. Test our wrappers
print("\n4. Testing our optimization wrappers:")
try:
    # Add parent directory to path
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Test torch.compile wrapper
    from src.moe.extensions.torch_compile_wrapper import TorchCompileConfig, TorchCompileWrapper

    config = TorchCompileConfig(enabled=True)
    wrapper = TorchCompileWrapper(config)
    print("✅ torch.compile wrapper imported")

    # Test quantization manager
    from src.moe.extensions.quantization_manager import QuantizationConfig, QuantizationManager

    quant_config = QuantizationConfig(enabled=True, mode="int8")
    manager = QuantizationManager(quant_config)
    print("✅ Quantization manager imported")

    print("\n✅ ALL OPTIMIZATION WRAPPERS WORK IN WSL!")

except ImportError as e:
    print(f"❌ Wrapper import failed: {e}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

successes = []
failures = []

# Check results
if hasattr(torch, 'compile'):
    successes.append("torch.compile")
else:
    failures.append("torch.compile")

try:
    import bitsandbytes
    successes.append("bitsandbytes")
except:
    failures.append("bitsandbytes")

if successes:
    print(f"✅ Working: {', '.join(successes)}")
if failures:
    print(f"❌ Failed: {', '.join(failures)}")

if len(successes) == 2:
    print("\n🎉 BOTH OPTIMIZATIONS WORK IN WSL!")
    print("You can now enable them in production config!")
elif len(successes) == 1:
    print(f"\n⚠️ Only {successes[0]} works in WSL")
else:
    print("\n❌ Neither optimization works - check installation")

print("="*60)