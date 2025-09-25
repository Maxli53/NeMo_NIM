# WSL2 + Project Migration to D: Drive - Final Steps

## Current Status
✅ **Phase 1-2 Complete**: Directories created, WSL shutdown
⏳ **Phase 3 In Progress**: WSL export running (15-30 min)

**Export Progress**: Check with PowerShell:
```powershell
Get-Item D:\WSL_Backup\ubuntu-24.04.tar | Select Name, @{N='GB';E={[math]::Round($_.Length/1GB,2)}}
```

---

## Remaining Steps (Run After Export Completes)

### Phase 4: Copy Project Files to D:
```powershell
# In PowerShell
xcopy "C:\Users\maxli\PycharmProjects\PythonProject\AI_agents" "D:\AI_Projects\NeMo_GPT" /E /I /H /Y

# Verify
dir D:\AI_Projects\NeMo_GPT
```

### Phase 5: Unregister & Import WSL2
```powershell
# IMPORTANT: Export must be complete before running this!
# Check export is done:
Test-Path D:\WSL_Backup\ubuntu-24.04.tar  # Should return True

# Unregister (removes from C:)
wsl --unregister Ubuntu-24.04

# Import to D:
wsl --import Ubuntu-24.04 D:\WSL\Ubuntu D:\WSL_Backup\ubuntu-24.04.tar

# Set default
wsl --set-default Ubuntu-24.04
```

### Phase 6: Verify WSL2 Migration
```bash
# Start WSL
wsl

# Check location (should show D: drive)
df -h /

# Verify user directory
ls ~/
pwd
```

### Phase 7: Update Git Remote
```bash
# In WSL
cd /mnt/d/AI_Projects/NeMo_GPT

# Verify git repo
git status
git remote -v
```

### Phase 8: Restart Docker Container
```bash
# Remove old reference
docker rm nemo-gpt-oss 2>/dev/null || true

# Start with new D: paths
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -it -v /mnt/d/AI_Projects/NeMo_GPT/workspace:/workspace \
  --name nemo-gpt-oss -d nvcr.io/nvidia/nemo:25.07.gpt_oss bash

# Verify
docker ps
docker exec nemo-gpt-oss nvidia-smi
```

### Phase 9: Download GPT-OSS-20B (Finally!)
```bash
# In WSL
cd /mnt/d/AI_Projects/NeMo_GPT
mkdir -p models
cd models

# Download model (~14GB, 20-40 min)
git lfs install
git clone https://huggingface.co/openai/gpt-oss-20b
```

### Phase 10: Verify Complete Setup
```bash
# Check NeMo
docker exec nemo-gpt-oss python -c "from nemo.collections import llm; config = llm.GPTOSSConfig20B(); print(f'✓ NeMo: {config.num_moe_experts} experts, topk={config.moe_router_topk}')"

# Check model files
ls -lh /mnt/d/AI_Projects/NeMo_GPT/models/gpt-oss-20b/
```

---

## Quick Copy-Paste Script (After Export Done)

```powershell
# Phase 4: Copy project
xcopy "C:\Users\maxli\PycharmProjects\PythonProject\AI_agents" "D:\AI_Projects\NeMo_GPT" /E /I /H /Y

# Phase 5: Migrate WSL2
wsl --unregister Ubuntu-24.04
wsl --import Ubuntu-24.04 D:\WSL\Ubuntu D:\WSL_Backup\ubuntu-24.04.tar
wsl --set-default Ubuntu-24.04
```

```bash
# Phase 6-10: In WSL
cd /mnt/d/AI_Projects/NeMo_GPT
git status

# Restart container
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -it -v /mnt/d/AI_Projects/NeMo_GPT/workspace:/workspace \
  --name nemo-gpt-oss -d nvcr.io/nvidia/nemo:25.07.gpt_oss bash

# Download model
mkdir -p models && cd models
git clone https://huggingface.co/openai/gpt-oss-20b
```

---

## Post-Migration Cleanup (Optional)

After verifying everything works:

```powershell
# Remove old C: project copy
# rmdir /S "C:\Users\maxli\PycharmProjects\PythonProject\AI_agents"

# Remove backup (keep for safety initially)
# del D:\WSL_Backup\ubuntu-24.04.tar
```

---

## New Structure

```
D:\
├── WSL\Ubuntu\              # WSL2 system (~117GB)
├── WSL_Backup\
│   └── ubuntu-24.04.tar    # Backup (~50-80GB)
└── AI_Projects\
    └── NeMo_GPT\
        ├── models\
        │   └── gpt-oss-20b\  # ~14GB
        ├── workspace\
        ├── checkpoints\
        └── nemo\
```

---

## Troubleshooting

**If export fails:**
```powershell
wsl --shutdown
wsl --export Ubuntu-24.04 D:\WSL_Backup\ubuntu-24.04.tar
```

**If import fails:**
```powershell
# Check tar file exists
Test-Path D:\WSL_Backup\ubuntu-24.04.tar
# Re-run import
wsl --import Ubuntu-24.04 D:\WSL\Ubuntu D:\WSL_Backup\ubuntu-24.04.tar
```

**If Docker doesn't work:**
```bash
# Check Docker service
docker --version
sudo service docker status
sudo service docker start
```