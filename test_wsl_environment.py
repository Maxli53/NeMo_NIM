#!/usr/bin/env python3
"""
WSL Environment Verification Test
Confirms all optimizations work in WSL with PyCharm
"""

import sys
import torch
import time
import os

print("="*60)
print("WSL ENVIRONMENT VERIFICATION")
print("="*60)

# 1. Basic Environment Check
print("\n1. Environment Information:")
print(f"   Python: {sys.version}")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 2. Test torch.compile
print("\n2. Testing torch.compile:")
try:
    @torch.compile(mode="reduce-overhead")
    def compiled_fn(x, y):
        return torch.matmul(x, y).relu()

    x = torch.randn(512, 512).cuda()
    y = torch.randn(512, 512).cuda()

    # Warmup
    _ = compiled_fn(x, y)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        _ = compiled_fn(x, y)
    torch.cuda.synchronize()
    compiled_time = time.perf_counter() - start

    print(f"   ✅ torch.compile works")
    print(f"   Time for 100 iterations: {compiled_time:.3f}s")
except Exception as e:
    print(f"   ❌ torch.compile failed: {e}")

# 3. Test bitsandbytes
print("\n3. Testing bitsandbytes:")
try:
    import bitsandbytes as bnb

    # Create INT8 linear layer
    layer = bnb.nn.Linear8bitLt(512, 512, bias=False, has_fp16_weights=False).cuda()
    test_input = torch.randn(32, 512).cuda()
    output = layer(test_input)

    print(f"   ✅ Bitsandbytes works (v{bnb.__version__})")
    print(f"   INT8 layer output shape: {output.shape}")
except Exception as e:
    print(f"   ❌ Bitsandbytes failed: {e}")

# 4. Test our MoE optimizations
print("\n4. Testing MoE Optimizations:")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.moe.optimization_safety.optimization_control_center import get_control_center

    center = get_control_center()

    optimizations = [
        ("cuda_kernels", "Vectorized PyTorch"),
        ("async_io", "Async I/O Prefetching"),
        ("tiered_cache", "Tiered Caching"),
        ("torch_compile", "torch.compile JIT"),
        ("int8_weights", "INT8 Quantization")
    ]

    enabled_count = 0
    for opt_name, description in optimizations:
        is_enabled = center.is_optimization_enabled(opt_name)
        if is_enabled:
            enabled_count += 1
        status = "✅" if is_enabled else "❌"
        print(f"   {status} {opt_name:15} - {description}")

    print(f"\n   Total enabled: {enabled_count}/5")

except Exception as e:
    print(f"   ❌ Could not load optimization center: {e}")

# 5. Quick Performance Test
print("\n5. Quick Performance Test:")
try:
    # Test matrix multiplication performance
    size = 2048
    iterations = 50

    x = torch.randn(size, size).cuda()
    y = torch.randn(size, size).cuda()

    # Warmup
    for _ in range(5):
        _ = torch.matmul(x, y)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = torch.matmul(x, y)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tflops = (2 * size**3 * iterations) / (elapsed * 1e12)
    print(f"   Matrix multiply ({size}x{size}):")
    print(f"   Time: {elapsed:.3f}s for {iterations} iterations")
    print(f"   Performance: {tflops:.1f} TFLOPS")

except Exception as e:
    print(f"   ❌ Performance test failed: {e}")

# 6. Memory Test
print("\n6. Memory Usage:")
try:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved:  {reserved:.2f} GB")
        print(f"   Total:     {total:.2f} GB")
        print(f"   Available: {total - reserved:.2f} GB")
    else:
        print("   GPU not available")

except Exception as e:
    print(f"   ❌ Memory test failed: {e}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

all_working = True
checklist = [
    ("Python 3.12+", sys.version_info >= (3, 12)),
    ("PyTorch 2.5+", torch.__version__.startswith("2.5")),
    ("CUDA available", torch.cuda.is_available()),
    ("torch.compile", 'compiled_time' in locals()),
    ("bitsandbytes", 'bnb' in locals()),
    ("MoE optimizations", 'enabled_count' in locals() and enabled_count >= 5)
]

for item, status in checklist:
    icon = "✅" if status else "❌"
    print(f"{icon} {item}")
    if not status:
        all_working = False

if all_working:
    print("\n🎉 ALL SYSTEMS GO! WSL environment fully operational.")
    print("Ready to proceed with Dynamic Batching and Flash Attention.")
else:
    print("\n⚠️ Some components need attention.")

print("="*60)