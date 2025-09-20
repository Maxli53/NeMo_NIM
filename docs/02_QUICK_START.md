# Quick Start Guide

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)
- **RAM**: 32GB system memory recommended
- **Storage**: 100GB free space for model and cache
- **CUDA**: Version 11.8+ (12.1 recommended)

### Software Requirements
```bash
# Check Python version (3.11+ required)
python --version

# Check CUDA version
nvcc --version

# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourorg/gpt-oss-moe.git
cd gpt-oss-moe
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv_gptoss
venv_gptoss\Scripts\activate

# Linux/Mac
python -m venv venv_gptoss
source venv_gptoss/bin/activate
```

### 3. Install Dependencies
```bash
# Core dependencies
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers safetensors accelerate
pip install numpy pyyaml tqdm

# Optional optimizations
pip install triton  # For CUDA kernel fusion
pip install prometheus-client  # For monitoring
```

### 4. Download GPT-OSS-20B Model
```bash
# Using the provided script
python download_gpt_oss.py

# Or using Hugging Face CLI
huggingface-cli download openai/gpt-oss-20b --local-dir ./gpt-oss-20b
```

## Basic Usage

### 1. Simple Inference (No Optimizations)
```python
from native_moe_complete import NativeMoE
from moe_config import MoEConfig

# Create default config (all optimizations OFF)
config = MoEConfig()
config.model_path = "./gpt-oss-20b"

# Initialize model
model = NativeMoE(config)

# Run inference
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
output = model.forward(input_ids)
print(f"Output shape: {output.shape}")
```

### 2. Enable Safe Optimizations
```python
from moe_config import MoEConfig

# Create config with safe optimizations
config = MoEConfig()

# Enable tiered caching (safest optimization)
config.cache.mode = "tiered"
config.cache.gpu_capacity_gb = 2.0
config.cache.ram_capacity_gb = 16.0

# Enable async I/O (safe with fallback)
config.async_io.enabled = True
config.async_io.fallback_to_sync = True

# Initialize model with optimizations
model = NativeMoE(config)
```

### 3. Production Configuration
```python
# Load from YAML configuration
from moe_config import MoEConfig

config = MoEConfig.from_yaml("configs/production.yaml")

# Or create programmatically
config = MoEConfig()
config.cuda_kernels.enabled = torch.cuda.is_available()
config.async_io.enabled = True
config.cache.mode = "tiered"
config.multi_gpu.enabled = torch.cuda.device_count() > 1

# All have safety fallbacks
config.cuda_kernels.fallback_on_error = True
config.async_io.fallback_to_sync = True
config.multi_gpu.fallback_single_gpu = True

model = NativeMoE(config)
```

## Running Tests

### 1. Quick Validation
```bash
# Run basic tests
python test_moe_checklist.py

# Expected output:
# ✅ Expert Mixing Tests: 36/36 passed
# ✅ Edge Cases: 12/12 passed
# ✅ Forward Pass: 12/12 passed
```

### 2. Comprehensive Test Suite
```bash
# Run full test suite
python test_suite_v3.py

# Run optimization tests
python test_optimizations.py
```

### 3. Performance Benchmark
```python
from native_moe_complete import benchmark_performance

# Run benchmark
results = benchmark_performance(
    batch_sizes=[1, 2, 4, 8],
    seq_lengths=[128, 256, 512],
    num_iterations=100
)

print(f"Throughput: {results['throughput']} tokens/sec")
print(f"Latency P50: {results['latency_p50']}ms")
print(f"Memory Usage: {results['memory_gb']}GB")
```

## Multi-Agent System

### 1. Basic Multi-Agent Discussion
```python
from phase4_multi_agent import run_multi_agent_discussion

# Run discussion on a task
result = run_multi_agent_discussion(
    task="Design a bio-inspired quantum computer",
    max_rounds=5
)

print(f"Consensus reached: {result['consensus']}")
print(f"Final synthesis: {result['synthesis']}")
```

### 2. With Real GPT-OSS Model
```python
from multi_agent_integration import MultiAgentWithGPTOSS

# Initialize with real model
system = MultiAgentWithGPTOSS(
    model_path="./gpt-oss-20b",
    config=config
)

# Run discussion
result = system.discuss(
    "Create an energy storage system using MoE principles"
)
```

## Monitoring

### 1. Start Prometheus Metrics Server
```python
from monitoring import start_metrics_server

# Start on port 8000
start_metrics_server(port=8000)

# Metrics available at http://localhost:8000
```

### 2. View Real-time Dashboard
```bash
# Start Streamlit dashboard
streamlit run dashboard.py

# Access at http://localhost:8501
```

## Common Commands

```bash
# Check GPU memory usage
nvidia-smi

# Monitor in real-time
watch -n 1 nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Run with specific GPU
CUDA_VISIBLE_DEVICES=0 python your_script.py

# Profile performance
python -m torch.profiler your_script.py
```

## Configuration Examples

### Minimal Memory Configuration
```yaml
# configs/minimal_memory.yaml
cache:
  mode: single
  gpu_capacity_gb: 1.0

optimizations:
  gradient_checkpointing: true
  memory_pools: false

quantization:
  int8_weights: true
  int8_activations: false
```

### Maximum Performance Configuration
```yaml
# configs/max_performance.yaml
cuda_kernels:
  enabled: true

async_io:
  enabled: true
  prefetch_window: 5
  max_concurrent_loads: 16

cache:
  mode: tiered
  gpu_capacity_gb: 4.0
  ram_capacity_gb: 32.0

multi_gpu:
  enabled: true
  world_size: 4
```

### Safe Production Configuration
```yaml
# configs/production_safe.yaml
# All risky optimizations disabled
cuda_kernels:
  enabled: false  # Enable after testing

async_io:
  enabled: true
  fallback_to_sync: true

cache:
  mode: tiered  # Safe and effective

multi_gpu:
  enabled: false  # Enable after validation

quantization:
  int8_weights: false  # Test first
  int4_experimental: false  # Never in production initially
```

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
config.max_batch_size = 2

# Reduce cache size
config.cache.gpu_capacity_gb = 1.0

# Enable gradient checkpointing
config.optimizations.gradient_checkpointing = True

# Clear cache
torch.cuda.empty_cache()
```

### Slow Performance
```python
# Check if optimizations are enabled
print(config.get_active_optimizations())

# Enable safe optimizations
config.async_io.enabled = True
config.cache.mode = "tiered"
```

### Model Not Loading
```bash
# Verify model files exist
ls -la gpt-oss-20b/

# Check file integrity
python validate_model.py --path gpt-oss-20b
```

## Next Steps

1. **Explore Optimizations**: See [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)
2. **Deploy to Production**: See [50_DEPLOYMENT.md](docs/DEPLOYMENT.md)
3. **Run Benchmarks**: See [31_PERFORMANCE_BENCHMARKS.md](31_PERFORMANCE_BENCHMARKS.md)
4. **Integrate Multi-Agent**: See [40_MULTI_AGENT_OVERVIEW.md](40_MULTI_AGENT_OVERVIEW.md)

## Support

- GitHub Issues: [Report bugs](https://github.com/yourorg/gpt-oss-moe/issues)
- Documentation: [Full docs](00_DOCUMENTATION_INDEX.md)
- Community: [Discord/Slack]

---

*For detailed implementation, see [11_NATIVE_MOE_IMPLEMENTATION.md](11_NATIVE_MOE_IMPLEMENTATION.md)*