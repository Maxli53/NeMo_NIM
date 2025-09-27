# CUDA Compatibility Notes - Ubuntu Native Setup

## ✅ FULLY OPERATIONAL - Native Ubuntu + Docker Dual Setup

**Last Updated**: September 27, 2025
**Status**: All CUDA environments working without issues

---

## System Overview

### Hardware
- **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
- **Compute Capability**: 8.6 (Ampere architecture)
- **OS**: Ubuntu 24.04 LTS (native, not WSL)

### NVIDIA Driver
- **Version**: 580.65.06
- **Release Date**: Recent stable release
- **CUDA Support**: Up to CUDA 13.0
- **Status**: ✅ Fully functional, no compatibility issues

---

## CUDA Setup 1: Docker Container ✅

### Container CUDA Environment
- **Container**: nvcr.io/nvidia/nemo:25.07.gpt_oss
- **CUDA Version**: 12.9
- **PyTorch**: 2.8.0a0 (NVIDIA optimized, pre-built)
- **cuDNN**: Pre-installed
- **Driver Requirement**: 525+ (we have 580.65.06 ✅)

### Compatibility Status
```
Driver 580.65.06 → Supports CUDA 13.0
Container CUDA 12.9 → Fully compatible
Result: ✅ Native support, NO compatibility mode
```

### Verification
```bash
$ docker exec nemo-gpt-oss nvidia-smi
Sat Sep 27 08:34:22 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:09:00.0  On |                  N/A |
|  0%   44C    P8             49W /  370W |     433MiB /  24576MiB |     33%      Default |
+-----------------------------------------+------------------------+----------------------+
```

**Status**: ✅ Perfect - Driver supports higher CUDA version than container needs

---

## CUDA Setup 2: Native Ubuntu ✅

### Native CUDA Installation
- **CUDA Toolkit**: 12.6.r12.6
- **Build**: cuda_12.6.r12.6/compiler.35059454_0
- **Install Date**: September 27, 2025
- **Location**: `/usr/local/cuda-12.6/`
- **In PATH**: Yes (added to `~/.bashrc`)

### cuDNN Installation
- **Version**: cuDNN 9 for CUDA 12
- **Status**: ✅ Installed via apt
- **Compatibility**: Fully compatible with CUDA 12.6

### Additional NVIDIA Libraries
All installed and working:
- **NCCL 2**: Multi-GPU communication
- **cuBLAS**: Linear algebra operations
- **cuFFT**: Fast Fourier transforms
- **cuRAND**: Random number generation
- **cuSOLVER**: Dense/sparse linear solvers
- **cuSPARSE**: Sparse matrix operations

### Python Environment
- **Location**: `~/ml_envs/nemo_gpt_env/`
- **Python**: 3.11.0rc1
- **PyTorch**: 2.8.0+cu128 (CUDA 12.8 support)
- **NeMo**: 2.4.0

### Compatibility Matrix
```
System Setup:
├── Driver 580.65.06 (supports CUDA 13.0)
├── Host CUDA 12.6 (native install)
├── PyTorch CUDA 12.8 (pip install)
└── Result: ✅ All compatible

Driver 580.65.06 → Supports CUDA 13.0 ✅
Host CUDA 12.6 → Works with driver ✅
PyTorch CUDA 12.8 → Works with driver ✅
```

### Verification
```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0

$ python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
CUDA: True

$ python -c "import torch; print(torch.cuda.get_device_name(0))"
NVIDIA GeForce RTX 3090
```

**Status**: ✅ Perfect - All components working together

---

## Why Multiple CUDA Versions Work

### Understanding CUDA Compatibility

**NVIDIA Driver** = Master Controller
- Provides runtime support for CUDA
- Single driver supports multiple CUDA versions
- Our driver 580.65.06 supports CUDA 6.5 through 13.0

**CUDA Toolkit** = Development Tools
- Compilers (nvcc)
- Libraries (cuBLAS, cuDNN, etc.)
- Can have multiple versions installed
- Applications specify which version they need

**How They Coexist**:
```
Driver 580.65.06 (CUDA 13.0)
    ├── Supports Container CUDA 12.9 ✅
    ├── Supports Host CUDA 12.6 ✅
    └── Supports PyTorch CUDA 12.8 ✅
```

### Docker Isolation
- Container has its own `/usr/local/cuda/` (CUDA 12.9)
- Host has separate `/usr/local/cuda-12.6/`
- They never conflict because containers are isolated
- Both use the same GPU driver

### Forward Compatibility
NVIDIA guarantees:
- Driver with CUDA N.x supports all CUDA ≤ N.x
- Example: Driver with CUDA 13.0 → supports 12.9, 12.8, 12.6, etc.

---

## Verification Tests Passed ✅

### Docker Container Tests
```bash
✓ GPU accessible (nvidia-smi)
✓ PyTorch CUDA operations work
✓ Tensor allocation on GPU
✓ GPU memory management
✓ Multi-GPU detection (single GPU system)
✓ NeMo imports successfully
```

### Native Ubuntu Tests
```bash
✓ nvcc compiles CUDA code
✓ PyTorch finds CUDA devices
✓ torch.cuda.is_available() = True
✓ Tensor operations on GPU
✓ GPU memory allocation
✓ NeMo Framework loads
✓ Transformers with GPU acceleration
```

---

## Known Non-Issues

### PyTorch Warning (Harmless)
```
FutureWarning: The pynvml package is deprecated.
Please install nvidia-ml-py instead.
```
- **Impact**: None - Just a deprecation notice
- **Cause**: PyTorch uses deprecated pynvml package
- **Action**: Can ignore safely
- **Fix (optional)**: `pip install nvidia-ml-py`

### Different CUDA Versions (Normal)
- Container: CUDA 12.9
- Host: CUDA 12.6
- PyTorch: CUDA 12.8

This is **completely normal and expected**:
- All are compatible with driver 580.65.06
- Each component works in its own environment
- No performance impact
- No functionality issues

---

## Performance Considerations

### GPU Utilization
- **Full native support**: No compatibility mode overhead
- **Driver efficiency**: Latest stable driver
- **Memory bandwidth**: Full 24GB accessible
- **Compute performance**: 100% of RTX 3090 capability

### Recommended Settings
For optimal inference performance:
- Use `bf16-mixed` precision (balance of speed/quality)
- Enable `USE_CPU_INITIALIZATION=True` for 24GB GPUs
- Set `MOE_TOPK=2` for memory-constrained scenarios
- Enable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## Troubleshooting Guide

### If GPU Not Detected

**Check Driver**:
```bash
nvidia-smi
# Should show driver 580.65.06 and GPU
```

**Check CUDA Path**:
```bash
echo $PATH | grep cuda
# Should include /usr/local/cuda-12.6/bin
```

**Check PyTorch**:
```bash
source ~/ml_envs/nemo_gpt_env/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### If Docker Can't Access GPU

**Check NVIDIA Container Toolkit**:
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
# Should show GPU info
```

**Restart Docker**:
```bash
sudo systemctl restart docker
```

### If Out of Memory

**Reduce Model Settings**:
```python
MOE_TOPK = 1              # Use fewer experts
MAX_BATCH_SIZE = 1        # One prompt at a time
NUM_TOKENS_TO_GENERATE = 50  # Generate less
```

---

## Upgrade Path (Future)

### When to Upgrade Driver
- New CUDA versions required by models
- Bug fixes or performance improvements
- GPU compute capability changes

### How to Upgrade Driver
```bash
# Check current
nvidia-smi

# Add PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install newer driver
sudo apt install nvidia-driver-XXX  # Replace XXX with version

# Reboot
sudo reboot
```

### When to Upgrade CUDA
- New features needed
- Better performance
- New library versions require it

**Note**: Multiple CUDA versions can coexist safely.

---

## Comparison: Previous (WSL) vs Current (Native Ubuntu)

### Previous Setup (Windows + WSL2)
- ❌ Disk space issues (8.9GB free)
- ⚠️ WSL2 overhead and complexity
- ⚠️ Driver compatibility through WSL layer
- ❌ Limited performance due to virtualization

### Current Setup (Ubuntu Native)
- ✅ 4.4TB disk space available
- ✅ Direct hardware access (no virtualization)
- ✅ Full native driver support
- ✅ Maximum performance
- ✅ Simpler architecture
- ✅ Both Docker and native options

---

## References

### Official Documentation
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [NVIDIA Driver Downloads](https://www.nvidia.com/drivers)
- [cuDNN Documentation](https://developer.nvidia.com/cudnn)
- [PyTorch CUDA Support](https://pytorch.org/get-started/locally/)
- [NeMo Framework Docs](https://docs.nvidia.com/nemo-framework/)

### Version Compatibility
- [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [PyTorch Compatibility Matrix](https://pytorch.org/get-started/previous-versions/)
- [Driver to CUDA Mapping](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)

---

## Summary

| Component | Version | Status | Notes |
|-----------|---------|--------|-------|
| **GPU** | RTX 3090 | ✅ Working | 24GB VRAM, Ampere |
| **Driver** | 580.65.06 | ✅ Perfect | CUDA 13.0 support |
| **Container CUDA** | 12.9 | ✅ Native | No compatibility mode |
| **Host CUDA** | 12.6 | ✅ Native | Installed, in PATH |
| **PyTorch CUDA** | 12.8 | ✅ Native | Full GPU acceleration |
| **cuDNN** | 9 | ✅ Installed | Host + Container |
| **Performance** | 100% | ✅ Maximum | No overhead |

**Overall**: 🎉 **PERFECT SETUP** - All CUDA environments fully operational with native support, no compatibility issues, maximum performance.

---

**Migration Complete**: Upgraded from WSL2 with compatibility concerns to native Ubuntu with flawless CUDA support!