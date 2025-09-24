# Official NeMo GPT-OSS Container
# Based on: https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/gpt_oss.html

FROM nvcr.io/nvidia/nemo:25.07.gpt_oss

# Set working directory
WORKDIR /workspace

# Install additional dependencies if needed
RUN pip install --no-cache-dir \
    nemo-run \
    nvidia-modelopt \
    tensorrt-llm

# Copy project files
COPY ./workspace /workspace
COPY ./scripts /scripts
COPY ./configs /configs

# Set environment variables
ENV PYTHONPATH="/workspace:${PYTHONPATH}"
ENV CUDA_VISIBLE_DEVICES=0
ENV NEMO_CACHE_DIR="/workspace/.cache"

# Create necessary directories
RUN mkdir -p /workspace/.cache /workspace/logs

# Default command
CMD ["/bin/bash"]