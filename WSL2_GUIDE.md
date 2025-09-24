# 🚀 WSL2 Complete Setup Guide for NeMo + NIM

## Quick Start (2 Steps!)

### Step 1: Windows Setup (Run Once)
```powershell
# In PowerShell as Administrator:
.\setup_wsl2_windows.ps1
# Restart computer when prompted
```

### Step 2: Start Development
```batch
# After restart, just double-click:
start_wsl2_dev.bat
```

That's it! Everything will be installed automatically.

## 🎯 What Gets Installed

### In Windows:
- ✅ WSL2 with Ubuntu
- ✅ GPU support enabled
- ✅ Memory configured (24GB)

### In WSL2:
- ✅ Docker with GPU support
- ✅ NVIDIA Container Toolkit
- ✅ NeMo + NIM container
- ✅ All frameworks pre-installed

## 📋 Development Workflow

### 1. Enter WSL2
```batch
# From Windows CMD/PowerShell:
wsl

# Or use the batch file:
start_wsl2_dev.bat
```

### 2. Quick Commands in WSL2
```bash
# These work from anywhere after setup:
nemo-enter    # Enter Docker container
nemo-train    # Run training
nemo-deploy   # Deploy model
nemo-jupyter  # Start Jupyter
nemo-gpu      # Check GPU status
nemo-logs     # View logs
ai            # Go to project directory
```

### 3. Inside Docker Container
```bash
# After running nemo-enter:
python test_installation.py              # Test setup
python train.py --data data/sample.jsonl # Train model
python deploy.py --model models/checkpoint # Deploy
jupyter lab --ip=0.0.0.0 --allow-root   # Jupyter
```

## 🔧 WSL2 Configuration

### Memory & CPU (.wslconfig)
Already configured at `C:\Users\%USERNAME%\.wslconfig`:
```ini
[wsl2]
memory=24GB      # Adjust based on your RAM
processors=8     # Adjust based on your CPU
swap=8GB
localhostForwarding=true

[experimental]
sparseVhd=true
autoMemoryReclaim=gradual
```

### GPU Support
- **Windows 11**: Works out of box
- **Windows 10**: Need build 21H2+
- **Driver**: Download from https://developer.nvidia.com/cuda/wsl

## 📊 Resource Usage in WSL2

| Component | Memory | Disk | Notes |
|-----------|--------|------|-------|
| WSL2 Ubuntu | 2GB | 5GB | Base system |
| Docker | 2GB | 1GB | Docker daemon |
| NeMo Container | 8-16GB | 15GB | When running |
| Model Loading | 10-20GB | 40GB | GPT-OSS-20B |

## 🌐 Accessing Services from Windows

All services are accessible from Windows browsers:

| Service | URL | Purpose |
|---------|-----|---------|
| Jupyter Lab | http://localhost:8888 | Notebooks |
| FastAPI | http://localhost:8000/docs | API docs |
| TensorBoard | http://localhost:6006 | Training metrics |
| NIM Server | http://localhost:8080 | Inference |

## ⚡ Performance Tips

### 1. File System Performance
```bash
# FAST: Keep project files in WSL2
/home/username/projects/

# SLOW: Accessing Windows files
/mnt/c/Users/...

# For best performance, clone directly in WSL2:
cd ~
git clone <your-repo>
```

### 2. Docker Performance
```bash
# Use WSL2 backend (automatic)
docker info | grep "Storage Driver"
# Should show: overlay2

# Prune regularly
docker system prune -a
```

### 3. GPU Optimization
```bash
# Check GPU in WSL2
nvidia-smi

# Check GPU in Docker
docker exec nemo-nim-dev nvidia-smi

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## 🐛 Troubleshooting

### "WSL2 not installed"
```powershell
# In PowerShell as Admin:
wsl --install
wsl --set-default-version 2
```

### "No GPU in WSL2"
```powershell
# Update WSL2:
wsl --update

# Check Windows NVIDIA driver:
nvidia-smi
# Must be 510.06 or newer
```

### "Docker not starting in WSL2"
```bash
# In WSL2:
sudo service docker start

# Or restart Docker:
sudo service docker restart
```

### "Out of memory"
```powershell
# Adjust in C:\Users\%USERNAME%\.wslconfig
[wsl2]
memory=16GB  # Reduce if needed

# Then restart WSL2:
wsl --shutdown
wsl
```

### "Slow file access"
```bash
# Move project to WSL2 filesystem:
cp -r /mnt/c/Users/maxli/PycharmProjects/PythonProject/AI_agents ~/ai_agents
cd ~/ai_agents
```

## 🔄 Daily Workflow

### Morning Setup
```bash
# 1. Start WSL2
start_wsl2_dev.bat

# 2. Enter container
nemo-enter

# 3. Start Jupyter (optional)
nemo-jupyter
```

### During Development
```bash
# Edit in VS Code with WSL2 extension
code .

# Run training
nemo-train --data data/train.jsonl

# Monitor GPU
nemo-gpu

# View logs
nemo-logs
```

### End of Day
```bash
# Save work
git add .
git commit -m "Daily progress"
git push

# Stop containers
nemo-stop

# Exit WSL2
exit
```

## 💡 VS Code Integration

1. Install "Remote - WSL" extension
2. In WSL2: `code .`
3. VS Code opens with full WSL2 integration
4. Terminal runs in WSL2
5. Full IntelliSense and debugging

## 📚 Additional Resources

- [WSL2 Documentation](https://docs.microsoft.com/en-us/windows/wsl/)
- [Docker in WSL2](https://docs.docker.com/desktop/windows/wsl/)
- [NVIDIA CUDA on WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [VS Code with WSL2](https://code.visualstudio.com/docs/remote/wsl)

## ✅ Verification Checklist

After setup, verify everything works:

```bash
# In WSL2:
[ ] nvidia-smi                     # GPU visible
[ ] docker --version                # Docker installed
[ ] docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi  # GPU in Docker
[ ] docker-compose --version        # Docker Compose ready
[ ] nemo-enter                      # Can enter container
[ ] docker exec nemo-nim-dev python -c "import torch; print(torch.cuda.is_available())"  # PyTorch GPU

# All checked? You're ready to develop!
```

---

**Your API Key is already configured in `.env`!** Just run the setup scripts and start developing.