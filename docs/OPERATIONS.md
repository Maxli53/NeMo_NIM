# Operations Guide

## Environment Setup

### System Requirements

#### Hardware
- **GPU**: NVIDIA RTX 3090 or better (24GB VRAM minimum)
- **RAM**: 32GB system memory recommended
- **Storage**: 50GB free space for model and dependencies

#### Software
- **OS**: WSL2 or Linux (Ubuntu 20.04/22.04)
  - ⚠️ Windows native has torch.compile issues
- **CUDA**: 12.8 or newer
- **cuDNN**: 9.10.2
- **Python**: 3.10 - 3.12

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

### Required Dependencies

```txt
# requirements.txt
torch==2.8.0
transformers==4.56.2
safetensors==0.6.2
accelerate==1.10.1
bitsandbytes==0.47.0  # For INT8 (optional)
triton==3.4.0
numpy<2.0
```

## Production Configuration

### Default Settings (Recommended)

```python
# configs/production.yaml
model:
  path: "gpt-oss-20b/original"
  precision: "fp16"
  num_experts: 32
  experts_per_token: 4

optimization:
  sdpa: true           # Flash Attention
  torch_compile: false # DO NOT ENABLE - causes regression
  int8: false         # Too slow currently
  mixed_precision: false  # Overhead at batch=1

inference:
  batch_size: 1
  max_sequence_length: 128
  temperature: 0.7

monitoring:
  log_level: "INFO"
  metrics_interval: 60  # seconds
  health_check: true
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

### Pre-flight Check

```bash
# Verify environment is ready
python scripts/preflight_check.py

# Expected output:
# ✅ CUDA: Available (v12.8)
# ✅ GPU: NVIDIA GeForce RTX 3090
# ✅ VRAM: 25.8 GB
# ✅ PyTorch: 2.8.0+cu128
# ✅ All dependencies installed
```

### Production Inference

```bash
# Basic inference
python main.py \
  --model gpt-oss-20b/original \
  --fp16 \
  --sdpa \
  --top-k 4

# With monitoring
python main.py \
  --model gpt-oss-20b/original \
  --fp16 \
  --sdpa \
  --top-k 4 \
  --monitor \
  --log-level INFO
```

### Performance Testing

```bash
# Run benchmark
python tests/test_performance.py

# Expected performance:
# Throughput: 29 tokens/sec
# Memory: 7.3 GB
# First token: 30ms
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

### Health Check Thresholds

```yaml
thresholds:
  max_latency_ms: 500
  min_throughput_tps: 6
  max_memory_gb: 22
  max_gpu_temp_c: 85
  max_error_rate: 0.01
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

#### 3. dtype Mismatch (INT8)
```
Error: mat1 and mat2 must have the same dtype
Status: Known issue with INT8
Workaround: Use FP16 only
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

### Feature Flags

```python
from src.moe.optimization_safety import OptimizationControl

control = OptimizationControl()

# Safe optimizations (recommended)
control.enable("fp16")      # ✅ Use
control.enable("sdpa")      # ✅ Use
control.enable("top_k", 4)  # ✅ Use

# Problematic optimizations
control.disable("torch_compile")  # ❌ Makes things worse
control.disable("int8")          # ⚠️ Too slow
control.disable("mixed_precision") # ⚠️ Overhead > benefit
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

## Deployment Checklist

- [ ] WSL2/Linux environment (not Windows native)
- [ ] CUDA 12.8+ installed and verified
- [ ] Python environment with correct dependencies
- [ ] Model files downloaded (~13GB)
- [ ] Environment variables set
- [ ] Pre-flight check passed
- [ ] Benchmark meets targets (>6 TPS)
- [ ] Monitoring enabled
- [ ] torch.compile DISABLED
- [ ] SDPA enabled
- [ ] Top-k=4 configured

## Support

For issues, check:
1. [PERFORMANCE.md](PERFORMANCE.md) for optimization status
2. [TECHNICAL.md](TECHNICAL.md) for implementation details
3. GitHub Issues for known problems