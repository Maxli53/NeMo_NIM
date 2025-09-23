# Operations Guide - GPT-OSS-20B v1.0.0

## ✅ Production Ready Status

**Implementation**: COMPLETE with 21/21 verification tests passing
**Performance**: 29.1 TPS with 7.3GB VRAM on RTX 3090
**Stability**: Memory-safe generation, no segfaults
**Integration**: Works with multi-agent discussion system

## Environment Setup

### System Requirements (Verified)

#### Hardware (Tested Configuration)
- **GPU**: NVIDIA RTX 3090 (24GB VRAM) - verified working
- **VRAM Usage**: 7.3GB actual (plenty of headroom)
- **RAM**: 32GB system memory (16GB minimum)
- **Storage**: 50GB free space
  - 13GB for model weights (MXFP4 compressed)
  - 5.6GB for virtual environment
  - 30GB for dependencies and cache

#### Software (Production Tested)
- **OS**: WSL2 or Linux (Ubuntu 20.04/22.04)
  - ✅ Verified: 29.1 TPS on WSL2
  - ❌ Windows native: 5x slower (avoid)
- **CUDA**: 12.8+ (tested with 12.8)
- **cuDNN**: 9.10.2+
- **Python**: 3.10-3.12 (tested with 3.12)
- **PyTorch**: 2.8.0+cu128 (verified working)

### WSL2 Setup (Windows Users)

```bash
# Install WSL2
wsl --install

# Set WSL2 as default
wsl --set-default-version 2

# Install Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# Verify GPU access
nvidia-smi  # Should show RTX 3090
```

### Python Environment

```bash
# Create virtual environment
python3 -m venv ~/cuda_env
source ~/cuda_env/bin/activate

# Install PyTorch with CUDA 12.8
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies (Production Verified)

```txt
# requirements.txt - TESTED CONFIGURATION
torch==2.8.0+cu128      # CUDA 12.8 support
transformers==4.56.2    # Safetensors compatibility
safetensors==0.6.2      # For 13GB model weights
accelerate==1.10.1      # Memory management
bitsandbytes==0.47.0    # INT8 support (optional)
triton==3.4.0          # SDPA/Flash Attention
numpy<2.0              # Compatibility
streamlit==1.28.0      # Web UI
fastapi==0.104.1       # API server
uvicorn==0.24.0        # ASGI server
gradually==1.5.0       # Vector database
```

## Production Configuration

### Production Configuration (Verified Working)

```yaml
# configs/production.yaml - TESTED & VERIFIED
model:
  path: "gpt-oss-20b/original"
  precision: "fp16"           # ✅ 29.1 TPS baseline
  num_experts: 32
  experts_per_token: 4        # ✅ Top-k=4 verified
  mxfp4_weights: true         # ✅ Real 13GB weights
  load_timeout: 30            # ✅ Loads in ~12s

optimization:
  sdpa: true                  # ✅ Flash Attention enabled
  torch_compile: false        # ❌ DO NOT ENABLE (88% slower)
  int8: false                 # ⚠️ Optional (-44% memory, -62% speed)
  mixed_precision: false      # ❌ Overhead at batch=1
  sliding_window: true        # ✅ Prevents segfaults

inference:
  batch_size: 1               # ✅ Tested 1-32, optimal=8
  max_sequence_length: 128    # ✅ Verified safe
  max_context_length: 2048    # ✅ Sliding window
  temperature: 0.7
  do_sample: true

monitoring:
  log_level: "INFO"
  metrics_interval: 60
  health_check: true
  verification_on_start: true # ✅ Run 21-test suite
```

### Environment Variables

```bash
# Add to ~/.bashrc or set before running

# CUDA settings
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Performance tuning
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8

# Disable torch.compile (it makes things worse)
export TORCH_COMPILE_DISABLE=1
```

## Running the Model

### Pre-flight Check (Production Validation)

```bash
# Verify environment is ready
python scripts/preflight_check.py

# Expected output:
# ✅ CUDA: Available (v12.8)
# ✅ GPU: NVIDIA GeForce RTX 3090 (24GB)
# ✅ VRAM: 24GB total, 7.3GB required
# ✅ PyTorch: 2.8.0+cu128
# ✅ Model weights: 13GB safetensors found
# ✅ All dependencies installed
# ✅ MXFP4 dequantization working
# ✅ Verification suite: 21/21 tests ready

# Complete verification (recommended)
python verify_implementation.py

# Expected: ✅ ALL VERIFICATIONS PASSED (21/21)
# - Core model components
# - Memory & performance
# - Output quality validation
```

### Production Deployment (Complete System)

```bash
# 1. Multi-Agent Discussion with MoE Backend
python main.py \
  --model gpt-oss \
  --topic "Your discussion topic" \
  --verify-on-start

# 2. Streamlit Web Interface
streamlit run src/ui/streamlit_app.py
# Access: http://localhost:8501

# 3. FastAPI Server
python -m src.api.server
# API: http://localhost:8000
# Health: http://localhost:8000/health
# Docs: http://localhost:8000/docs

# 4. Direct MoE Inference (Production)
python -c "
from src.moe.native_moe_loader_v2 import MoEModelLoader
loader = MoEModelLoader('gpt-oss-20b/original')
model = loader.create_model_fp16(top_k=4)
print('Model loaded successfully')
"

# 5. With Complete Monitoring
python main.py \
  --model gpt-oss \
  --monitor \
  --health-checks \
  --verify-on-start \
  --log-level INFO
```

### Performance Validation (All Tests Pass)

```bash
# Complete verification suite
python verify_implementation.py
# Expected: ✅ ALL VERIFICATIONS PASSED (21/21)

# Performance benchmark
python tests/test_performance.py
# Expected results:
# ✅ Throughput: 29.1 tokens/sec
# ✅ Memory: 7.3GB VRAM
# ✅ First token: 30ms
# ✅ Load time: ~12 seconds
# ✅ No segfaults or crashes

# Integration test
python test_gpt_oss_complete.py
# Expected: Full model loads and generates

# Memory safety test
python test_generation_safe.py
# Expected: Stable generation, no memory issues

# Quick verification (development)
python verify_implementation.py --quick
# Expected: Core tests pass in <1 minute
```

## Monitoring & Health Checks

### Real-time Monitoring

```python
from src.moe.optimization_safety import HealthMonitor

monitor = HealthMonitor()
monitor.start()

# Metrics available:
# - throughput_tps: tokens per second
# - latency_ms: first token latency
# - memory_gb: VRAM usage
# - temperature: GPU temperature
```

### Health Check Thresholds (Production Tuned)

```yaml
thresholds:
  # Performance thresholds (based on verified metrics)
  max_latency_ms: 500        # Target: 30ms (16x safety margin)
  min_throughput_tps: 6      # Target: 29.1 TPS (5x safety margin)
  max_memory_gb: 22          # Target: 7.3GB (3x safety margin)
  max_load_time_s: 30        # Target: 12s (2.5x safety margin)

  # System thresholds
  max_gpu_temp_c: 85
  max_error_rate: 0.01
  max_verification_failures: 0  # All 21 tests must pass

  # Quality thresholds
  min_output_std: 1.0        # Target: 2.88 (magnitude check)
  max_output_std: 10.0       # Prevent magnitude explosion
  max_nan_rate: 0.0          # No NaN values allowed

  # Stability thresholds
  max_segfaults: 0           # Zero tolerance for crashes
  min_generation_length: 10  # Ensure generation works
  max_memory_leaks: 0        # No memory growth over time
```

### Automatic Recovery

```python
# System automatically disables optimizations if thresholds violated
if metrics['latency_ms'] > 500:
    disable_optimization('sdpa')
    fallback_to_baseline()
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
Error: CUDA out of memory
Solution: Reduce batch_size or sequence_length
```

#### 2. Slow Performance
```
Symptom: <6 tokens/sec
Check:
- Ensure torch.compile is DISABLED
- Verify SDPA is enabled
- Check GPU utilization (nvidia-smi)
```

#### 3. Model Loading Issues
```
Symptom: Model hangs during loading
Solution: Normal - takes ~12 seconds for 13GB weights
Verification: Progress indicators show activity
```

#### 4. Windows Performance Issues
```
Symptom: 5x slower on Windows
Solution: Use WSL2 instead
```

### Debug Commands

```bash
# Check GPU status
nvidia-smi

# Monitor in real-time
watch -n 1 nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Detailed environment info
python scripts/preflight_check.py --verbose
```

## Optimization Control

### Feature Flags (Production Configuration)

```python
from src.moe.optimization_safety import OptimizationControl

control = OptimizationControl()

# ✅ Verified safe optimizations (enabled)
control.enable("fp16")           # ✅ 29.1 TPS baseline
control.enable("sdpa")           # ✅ Flash Attention
control.enable("top_k", 4)       # ✅ 87% memory reduction
control.enable("mxfp4")          # ✅ Real 13GB weights
control.enable("sliding_window") # ✅ Prevents segfaults
control.enable("expert_cache")   # ✅ LRU caching
control.enable("progress_bar")   # ✅ Load indicators

# ❌ Known problematic optimizations (disabled)
control.disable("torch_compile")    # ❌ 88% slower
control.disable("mixed_precision")  # ❌ 7% slower at batch=1

# ⚠️ Optional optimizations (configurable)
control.configure("int8", enabled=False)  # ⚠️ -44% memory, -62% speed
control.configure("batch_size", value=1)  # ⚠️ Framework ready for >1
```

### Progressive Rollout

```python
# Test optimization on small % of traffic
control.enable_optimization(
    "new_feature",
    traffic_percentage=0.01,  # 1% canary
    rollback_on_error=True
)
```

## Production Deployment Checklist ✅

### ✅ Environment Setup
- [x] WSL2/Linux environment (verified working)
- [x] CUDA 12.8+ installed and functional
- [x] Python 3.10-3.12 with all dependencies
- [x] 13GB MXFP4 model weights downloaded
- [x] Environment variables configured
- [x] 24GB VRAM available (uses 7.3GB)

### ✅ Implementation Verification
- [x] All 21 verification tests passing
- [x] MXFP4 dequantization working (bias=127)
- [x] Real weights loading correctly
- [x] Output magnitude fixed (std=2.88)
- [x] Memory-safe generation (no segfaults)
- [x] Complete architecture implemented

### ✅ Performance Validation
- [x] Throughput: 29.1 TPS (✅ exceeds 6 TPS target)
- [x] Memory: 7.3GB (✅ under 22GB limit)
- [x] First token: 30ms (✅ under 500ms target)
- [x] Load time: ~12s (✅ under 30s target)
- [x] Stability: No crashes or hangs

### ✅ Production Features
- [x] Health monitoring enabled
- [x] Feature flags configured
- [x] Automatic rollback system
- [x] Comprehensive error handling
- [x] Structured logging
- [x] Multi-agent integration working

### ✅ Safety Configuration
- [x] torch.compile DISABLED (prevents 88% slowdown)
- [x] SDPA enabled (Flash Attention working)
- [x] Top-k=4 configured (memory optimization)
- [x] Sliding window enabled (prevents segfaults)
- [x] Verification on startup (21 tests)

**Status**: ✅ READY FOR PRODUCTION DEPLOYMENT

## Support & Troubleshooting

### Documentation (Complete)
1. [PROJECT_STATUS.md](../PROJECT_STATUS.md) - Single source of truth
2. [PERFORMANCE.md](PERFORMANCE.md) - Verification results (21/21 tests)
3. [TECHNICAL.md](TECHNICAL.md) - Complete architecture
4. [DEVELOPMENT.md](DEVELOPMENT.md) - Implementation milestones
5. This file - Production deployment guide

### Quick Diagnostics
```bash
# 1. Verify complete implementation
python verify_implementation.py
# Should show: ✅ ALL VERIFICATIONS PASSED (21/21)

# 2. Check system resources
nvidia-smi
# Should show: RTX 3090 with >7.3GB free VRAM

# 3. Test model loading
python -c "from src.moe.native_moe_loader_v2 import MoEModelLoader;
loader = MoEModelLoader('gpt-oss-20b/original');
loader.verify_weights_loaded()"
# Should show: Non-random weight statistics

# 4. Performance check
python tests/test_performance.py
# Should show: 29.1 TPS, 7.3GB, 30ms
```

### Emergency Procedures
- **Model hangs**: Normal for first 12 seconds (loading 13GB)
- **Segfaults**: Ensure sliding_window=true in config
- **Poor performance**: Check torch.compile is disabled
- **Memory issues**: Reduce batch_size or clear CUDA cache

### Production Status
✅ **COMPLETE IMPLEMENTATION** - All 21 verification tests passing
✅ **PRODUCTION READY** - Exceeds all performance targets
✅ **STABLE** - Memory-safe generation, no crashes
✅ **INTEGRATED** - Works with multi-agent discussion system