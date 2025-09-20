# 🔧 Troubleshooting Guide

## Table of Contents
- [Common Issues](#common-issues)
- [Installation Problems](#installation-problems)
- [Model Loading Issues](#model-loading-issues)
- [Memory and Performance](#memory-and-performance)
- [API and Network Issues](#api-and-network-issues)
- [Agent and Discussion Problems](#agent-and-discussion-problems)
- [Database and Storage](#database-and-storage)
- [GPU and CUDA Issues](#gpu-and-cuda-issues)
- [Docker and Kubernetes](#docker-and-kubernetes)
- [Debugging Tools](#debugging-tools)
- [FAQ](#faq)

## Common Issues

### Issue: System Won't Start

**Symptoms:**
- Application crashes on startup
- Import errors
- Configuration errors

**Solutions:**

1. **Check Python version:**
```bash
python --version  # Should be 3.10+
```

2. **Verify dependencies:**
```bash
pip list | grep -E "torch|transformers|faiss"
pip install --upgrade -r requirements.txt
```

3. **Check configuration:**
```python
python -c "from src.config import config; config.validate()"
```

4. **Reset configuration:**
```bash
cp config.yaml.example config.yaml
```

### Issue: Slow Performance

**Symptoms:**
- Long response times
- High CPU/GPU usage
- Memory leaks

**Solutions:**

1. **Enable quantization:**
```python
# In config.yaml
model:
  local:
    quantization: "8bit"  # or "4bit" for more compression
```

2. **Reduce batch size:**
```python
embedding:
  local:
    batch_size: 16  # Reduce from default 32
```

3. **Clear cache:**
```bash
rm -rf cache/*
python -c "import torch; torch.cuda.empty_cache()"
```

## Installation Problems

### Issue: Dependency Conflicts

**Error:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages
```

**Solution:**
```bash
# Create fresh environment
python -m venv venv_fresh
source venv_fresh/bin/activate  # Windows: venv_fresh\Scripts\activate

# Install with specific versions
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.35.0
pip install faiss-cpu==1.7.4  # or faiss-gpu for GPU
```

### Issue: Missing System Libraries

**Error:**
```
OSError: libcudart.so.11.0: cannot open shared object file
```

**Solution (Linux):**
```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-11-8

# Set environment variables
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

**Solution (Windows):**
```powershell
# Download and install CUDA from NVIDIA website
# Add to PATH:
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
```

## Model Loading Issues

### Issue: Model Not Found

**Error:**
```
OSError: gpt-oss-20b is not a local folder and is not a valid model identifier
```

**Solution:**
```bash
# Download model
python scripts/download_models.py --model gpt-oss-20b

# Or manually download
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="gpt-oss/gpt-oss-20b",
    local_dir="./models/gpt-oss-20b",
    local_dir_use_symlinks=False
)
```

### Issue: Model Too Large for Memory

**Error:**
```
CUDA out of memory. Tried to allocate X GB
```

**Solutions:**

1. **Enable CPU offloading:**
```python
# config.yaml
model:
  local:
    device_map: "auto"
    max_memory:
      0: "20GB"  # GPU 0
      "cpu": "30GB"  # CPU RAM
```

2. **Use smaller model or quantization:**
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
```

3. **Reduce MoE experts:**
```python
# config.yaml
model:
  local:
    moe:
      active_experts: 16  # Reduce from 32
```

### Issue: Tokenizer Errors

**Error:**
```
ValueError: Tokenizer class GPTOSSTokenizer does not exist
```

**Solution:**
```python
# Use AutoTokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "./models/gpt-oss-20b",
    trust_remote_code=True,  # Allow custom tokenizer
    use_fast=True
)
```

## Memory and Performance

### Issue: Memory Leak

**Symptoms:**
- Gradually increasing memory usage
- System becomes unresponsive
- OOM killer activates

**Diagnosis:**
```python
# Monitor memory usage
import tracemalloc
import psutil

tracemalloc.start()

# Your code here

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)

# Check process memory
process = psutil.Process()
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

**Solutions:**

1. **Clear cache regularly:**
```python
import gc
import torch

# After each discussion
gc.collect()
torch.cuda.empty_cache()
```

2. **Limit session history:**
```python
# config.yaml
session:
  management:
    max_sessions: 100  # Limit active sessions
    cleanup_old_sessions: true
    retention_days: 7
```

### Issue: High CPU Usage

**Solution:**
```python
# Limit thread usage
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch
torch.set_num_threads(4)
```

## API and Network Issues

### Issue: Connection Refused

**Error:**
```
ConnectionError: HTTPConnectionPool(host='localhost', port=8000): Max retries exceeded
```

**Solutions:**

1. **Check if server is running:**
```bash
ps aux | grep "mcp_server\|main.py"
netstat -tlnp | grep 8000
```

2. **Check firewall:**
```bash
# Linux
sudo ufw allow 8000

# Windows
netsh advfirewall firewall add rule name="AI Agent API" dir=in action=allow protocol=TCP localport=8000
```

3. **Check binding address:**
```python
# config.yaml
mcp:
  host: "0.0.0.0"  # Listen on all interfaces
  port: 8000
```

### Issue: CORS Errors

**Error:**
```
Access to fetch at 'http://localhost:8000' from origin 'http://localhost:8501' has been blocked by CORS policy
```

**Solution:**
```python
# src/mcp_server.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue: Rate Limiting

**Error:**
```
429 Too Many Requests
```

**Solution:**
```python
# Implement exponential backoff
import time
import random

def retry_with_backoff(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)
    raise Exception("Max retries exceeded")
```

## Agent and Discussion Problems

### Issue: Agents Not Responding

**Symptoms:**
- Discussion hangs
- No messages generated
- Timeout errors

**Diagnosis:**
```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual agent
from src.agents.expert import ExpertAgent

agent = ExpertAgent("TestAgent", "test domain")
response = await agent.generate_response("Test prompt")
print(response)
```

**Solutions:**

1. **Increase timeout:**
```python
# config.yaml
discussion:
  timeout_per_round: 300  # Increase from 120
```

2. **Check model loading:**
```python
from src.core.model_manager import model_manager

info = model_manager.get_model_info()
print(f"Model loaded: {info['loaded']}")
print(f"Device: {info['device']}")
```

### Issue: Poor Discussion Quality

**Symptoms:**
- Generic responses
- No citations
- Low vote scores

**Solutions:**

1. **Add knowledge sources:**
```python
# Add PDFs to vector database
from src.utils.pdf_processor import PDFProcessor
from src.core.vector_db import vector_db_manager

processor = PDFProcessor()
chunks = processor.chunk_pdf("data/textbook.pdf")

vector_db = vector_db_manager.get_or_create_db("physics")
for chunk in chunks:
    embedding = embedding_manager.embed(chunk["text"])
    vector_db.add(embedding, chunk["text"], chunk["source"])
```

2. **Adjust temperature:**
```python
# config.yaml
agents:
  experts:
    - name: "PhysicsExpert"
      temperature: 0.8  # Increase for more creativity
```

3. **Improve prompts:**
```python
# Custom system prompt
system_prompt = """
You are an expert in {domain}. 
Provide detailed, technical responses with specific examples.
Cite sources when available.
Be creative but scientifically accurate.
"""
```

### Issue: Consensus Never Reached

**Solutions:**

1. **Adjust thresholds:**
```python
# config.yaml
agents:
  consensus:
    thresholds:
      novelty: 6.0  # Lower from 7.0
      feasibility: 5.0  # Lower from 6.0
```

2. **Increase max rounds:**
```python
discussion:
  max_rounds: 15  # Increase from 10
```

## Database and Storage

### Issue: Vector Database Errors

**Error:**
```
RuntimeError: Error in faiss::IndexIVFFlat::search_preassigned
```

**Solutions:**

1. **Rebuild index:**
```python
from src.core.vector_db import FAISSVectorDB

# Rebuild corrupted index
db = FAISSVectorDB()
db.rebuild_index()
db.save("data/indices/physics.idx")
```

2. **Use different index type:**
```python
# config.yaml
vector_db:
  faiss:
    index_type: "Flat"  # More stable than IVF
```

### Issue: Session Not Saving

**Error:**
```
PermissionError: [Errno 13] Permission denied: 'data/sessions/session.json'
```

**Solution:**
```bash
# Fix permissions
chmod -R 755 data/
chown -R $USER:$USER data/

# Windows
icacls data /grant Everyone:F /T
```

## GPU and CUDA Issues

### Issue: CUDA Not Available

**Diagnosis:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
```

**Solutions:**

1. **Install correct PyTorch version:**
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2. **Set CUDA_VISIBLE_DEVICES:**
```bash
export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
```

### Issue: GPU Memory Fragmentation

**Solution:**
```python
# Reset GPU memory
import torch
import gc

# Clear all cached memory
torch.cuda.empty_cache()
gc.collect()

# Reset GPU
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
```

### Issue: Multi-GPU Not Working

**Solution:**
```python
# Enable DataParallel
from torch.nn import DataParallel

if torch.cuda.device_count() > 1:
    model = DataParallel(model)
    
# Or use device_map for transformers
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "gpt-oss-20b",
    device_map="auto",  # Automatic multi-GPU
    max_memory={0: "20GB", 1: "20GB"}  # Per-GPU limits
)
```

## Docker and Kubernetes

### Issue: Docker Build Fails

**Error:**
```
ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully
```

**Solutions:**

1. **Increase Docker memory:**
```json
// Docker Desktop settings
{
  "memoryMiB": 8192,
  "cpus": 4
}
```

2. **Use multi-stage build:**
```dockerfile
# Build stage
FROM python:3.10 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Runtime stage
FROM python:3.10-slim
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*
```

### Issue: Kubernetes Pods Crashing

**Diagnosis:**
```bash
kubectl logs pod-name -n ai-agent
kubectl describe pod pod-name -n ai-agent
kubectl get events -n ai-agent
```

**Solutions:**

1. **Increase resource limits:**
```yaml
resources:
  limits:
    memory: "32Gi"  # Increase from 16Gi
    cpu: "8"  # Increase from 4
  requests:
    memory: "16Gi"
    cpu: "4"
```

2. **Add health checks:**
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60  # Increase for model loading
  periodSeconds: 30
  timeoutSeconds: 10
  failureThreshold: 3
```

### Issue: GPU Not Available in Container

**Solution for Docker:**
```bash
# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Run with GPU
docker run --gpus all your-image
```

**Solution for Kubernetes:**
```yaml
# Install NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# Request GPU in pod spec
resources:
  limits:
    nvidia.com/gpu: 1
```

## Debugging Tools

### Memory Profiling

```python
# profile_memory.py
from memory_profiler import profile
import tracemalloc

tracemalloc.start()

@profile
def memory_intensive_function():
    # Your code here
    pass

memory_intensive_function()

# Get top memory allocations
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

### Performance Profiling

```python
# profile_performance.py
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(20)
```

### Network Debugging

```bash
# Test API endpoint
curl -v http://localhost:8000/health

# Test WebSocket
wscat -c ws://localhost:8000/ws/discussion

# Monitor network traffic
tcpdump -i lo -n port 8000

# Check open connections
netstat -an | grep 8000
lsof -i :8000
```

### Log Analysis

```python
# analyze_logs.py
import re
from collections import Counter

with open('logs/discussion.log', 'r') as f:
    logs = f.read()

# Find errors
errors = re.findall(r'ERROR.*', logs)
print(f"Total errors: {len(errors)}")

# Find slow operations
slow_ops = re.findall(r'took ([0-9.]+)s', logs)
slow_ops = [float(t) for t in slow_ops if float(t) > 1.0]
print(f"Slow operations: {len(slow_ops)}")
print(f"Average time: {sum(slow_ops)/len(slow_ops):.2f}s")

# Most common warnings
warnings = re.findall(r'WARNING: (.*)', logs)
warning_counts = Counter(warnings)
for warning, count in warning_counts.most_common(5):
    print(f"{count}: {warning}")
```

## FAQ

### Q: Can I use this without a GPU?

**A:** Yes, but with limitations:
```python
# config.yaml
model:
  local:
    device: "cpu"
    quantization: "8bit"  # Reduce memory usage
    
performance:
  optimization:
    cpu_offload: true
```

### Q: How much RAM/VRAM do I need?

**A:** Minimum requirements:
- **CPU only**: 32GB RAM (64GB recommended)
- **GPU**: 24GB VRAM for full model, 16GB with 8-bit quantization, 8GB with 4-bit
- **System RAM**: 16GB minimum, 32GB recommended

### Q: Can I add custom agents?

**A:** Yes, create a new agent class:
```python
from src.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, name, domain):
        super().__init__(name, domain)
    
    async def respond(self, queue, task, history, session_log):
        # Your implementation
        pass
```

### Q: How do I export discussions?

**A:** Multiple formats supported:
```python
from src.core.session import SessionManager

session_manager = SessionManager()
session = session_manager.get_session("session_id")

# Export formats
session.export_json("export.json")
session.export_markdown("export.md")
session.export_pdf("export.pdf")
```

### Q: Can I use different models?

**A:** Yes, configure in config.yaml:
```yaml
model:
  provider: "openai"  # or "anthropic", "local"
  model_name: "gpt-4"  # or "claude-3", "llama-2-70b"
```

### Q: How do I monitor system health?

**A:** Use the monitoring endpoints:
```bash
# Health check
curl http://localhost:8000/health

# Metrics (Prometheus format)
curl http://localhost:8000/metrics

# System info
curl http://localhost:8000/api/v1/system/info
```

### Q: What if embeddings are slow?

**A:** Optimize embedding generation:
```python
# Use batch processing
embeddings = embedding_manager.embed_batch(
    texts=documents,
    batch_size=64,  # Increase batch size
    show_progress=True
)

# Enable caching
embedding:
  cache:
    enabled: true
    backend: "redis"  # Faster than disk
```

### Q: How do I reset everything?

**A:** Complete reset:
```bash
# Stop all services
docker-compose down
pkill -f "python main.py"

# Clear all data
rm -rf data/* cache/* logs/*
rm -rf ~/.cache/huggingface

# Reinstall
pip install --force-reinstall -r requirements.txt

# Re-download models
python scripts/download_models.py --clean
```

---

For additional support:
- GitHub Issues: https://github.com/yourorg/ai-multi-agent/issues
- Documentation: https://docs.ai-agent.example.com
- Community Discord: https://discord.gg/ai-agent