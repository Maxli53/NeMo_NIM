# Ubuntu Native Setup Guide - NeMo GPT-OSS-20B

## What's Already Done ✅

1. **System Setup**
   - Git installed
   - Docker installed and configured
   - NVIDIA Container Toolkit installed
   - GPU accessible from Docker

2. **Project Files**
   - Model downloaded: `models/gpt-oss-20b/` (~13GB)
   - Scripts updated for Ubuntu paths
   - OOM fixes applied to `inference.py`:
     - `MOE_TOPK` reduced to 2 (from 4)
     - `MAX_BATCH_SIZE` reduced to 1 (from 8)
     - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` added

3. **Scripts Created**
   - `install_cuda_stack.sh` - Install CUDA, cuDNN, NVIDIA libraries
   - `install_python_ml_stack.sh` - Install Python, PyTorch, NeMo, ML tools
   - `start_nemo_container.sh` - Start Docker container
   - `install_docker.sh` - Already used
   - `install_nvidia_toolkit.sh` - Already used

## Current Status ⏳

- **Docker Container**: Pulling `nvcr.io/nvidia/nemo:25.07.gpt_oss` (~36GB)
- **Native Install**: Not started yet

## Installation Options

### Option 1: Docker-Based (Recommended - Easiest)

Everything pre-installed in container. Just pull and run.

```bash
# Wait for docker pull to complete
docker images | grep nemo

# Start container
./start_nemo_container.sh

# Access container
docker exec -it nemo-gpt-oss bash

# Inside container, run inference
cd /workspace
python inference.py
```

### Option 2: Native Installation (Advanced)

Install everything directly on Ubuntu. Takes longer but gives more control.

#### Step 1: Install CUDA Stack (~30 min)
```bash
./install_cuda_stack.sh
source ~/.bashrc
nvcc --version  # Verify
```

#### Step 2: Install Python ML Stack (~20 min)
```bash
./install_python_ml_stack.sh

# Activate virtual environment
source /media/ubumax/WD_BLACK/AI_Projects/NeMo_GPT/venv/bin/activate

# Verify PyTorch with CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}')"
```

#### Step 3: Test Native NeMo
```bash
# Inside activated venv
cd /media/ubumax/WD_BLACK/AI_Projects/NeMo_GPT
python -c "from nemo.collections import llm; print('NeMo works!')"
```

## Quick Reference

### Check Docker Pull Progress
```bash
docker images | grep nemo
```

### System Info
```bash
nvidia-smi                    # GPU status
df -h /media/ubumax/WD_BLACK  # Disk space
docker ps                     # Running containers
```

### Container Commands
```bash
# Start
./start_nemo_container.sh

# Access
docker exec -it nemo-gpt-oss bash

# Stop
docker stop nemo-gpt-oss

# Remove
docker rm nemo-gpt-oss

# View logs
docker logs nemo-gpt-oss
```

### Native Python Commands
```bash
# Activate venv
source venv/bin/activate

# Deactivate
deactivate

# Check installed packages
pip list | grep -E "torch|nemo|transformers"
```

## File Structure

```
/media/ubumax/WD_BLACK/AI_Projects/NeMo_GPT/
├── models/
│   └── gpt-oss-20b/          # Downloaded model (~13GB)
├── workspace/
│   ├── inference.py          # Fixed for 24GB VRAM
│   ├── convert_to_nemo.py   # Updated paths
│   ├── gpt_oss_training.py
│   └── checkpoints/
├── install_cuda_stack.sh     # Install CUDA natively
├── install_python_ml_stack.sh # Install Python + ML libs
├── start_nemo_container.sh   # Start Docker container
└── venv/                     # Python virtual env (after install)
```

## What's Installed (Native Option)

### CUDA Stack
- CUDA Toolkit 12.6
- cuDNN 9
- NCCL (multi-GPU communication)
- All CUDA libraries (cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE)

### Python ML Stack
- Python 3.11
- PyTorch 2.x with CUDA 12.1
- NeMo Framework
- Transformers, Datasets, Tokenizers
- Accelerate, BitsAndBytes
- LangChain, LlamaIndex, vLLM
- Scientific: NumPy, Pandas, SciPy, Scikit-learn
- Visualization: Matplotlib, Seaborn
- Tools: Jupyter, Weights & Biases, TensorBoard

## Next Steps

1. **Wait for Docker pull** to complete (check with `docker images | grep nemo`)
2. **Test Docker option first** (easiest)
3. **If Docker works**, you're done!
4. **(Optional)** Install native stack for more flexibility

## Troubleshooting

### OOM Errors
- Already fixed in `inference.py`
- If still occurs, reduce `MOE_TOPK` to 1
- Or reduce `NUM_TOKENS_TO_GENERATE`

### Docker Won't Start
```bash
sudo systemctl restart docker
docker ps
```

### Native Install Issues
```bash
# Check CUDA
nvcc --version
nvidia-smi

# Check Python
source venv/bin/activate
python --version
pip list
```

### Model Not Found
Make sure paths match:
- Docker: `/models/gpt-oss-20b/`
- Native: `/media/ubumax/WD_BLACK/AI_Projects/NeMo_GPT/models/gpt-oss-20b/`

## System Requirements Met ✅

- ✅ GPU: RTX 3090 (24GB VRAM)
- ✅ Driver: 580.65.06 (CUDA 13.0 support)
- ✅ Disk: 4.4TB free
- ✅ RAM: 31GB
- ✅ OS: Ubuntu (native)