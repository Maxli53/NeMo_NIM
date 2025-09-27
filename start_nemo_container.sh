#!/bin/bash
# Start NeMo container with proper mounts

echo "Starting NeMo GPT-OSS Container..."

# Remove old container if exists
docker rm -f nemo-gpt-oss 2>/dev/null || true

# Start container with Ubuntu paths
docker run --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -it \
    -v /media/ubumax/WD_BLACK/AI_Projects/NeMo_GPT/workspace:/workspace \
    -v /media/ubumax/WD_BLACK/AI_Projects/NeMo_GPT/models:/models \
    --name nemo-gpt-oss \
    -d \
    nvcr.io/nvidia/nemo:25.07.gpt_oss \
    bash

echo ""
echo "✅ Container started!"
echo "Access with: docker exec -it nemo-gpt-oss bash"
echo "Check GPU: docker exec nemo-gpt-oss nvidia-smi"