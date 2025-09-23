# Multi-stage build for production MoE inference
FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_COMPILE_DISABLE=1 \
    CUDA_HOME=/usr/local/cuda \
    PROJECT_ROOT=/app

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create Python symlink
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY pyproject.toml .
COPY .env.example .env

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p logs .cache test_results

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command for inference server
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage with additional tools
FROM base AS development

# Install dev dependencies
RUN pip install --no-cache-dir ruff mypy pytest pytest-cov

# Copy test files
COPY tests/ ./tests/

# Override command for development
CMD ["bash"]

# Production stage (minimal)
FROM base AS production

# Run as non-root user
RUN useradd -m -u 1000 moe && chown -R moe:moe /app
USER moe

# Expose API port
EXPOSE 8000

# Final command
CMD ["python", "-m", "src.api.server"]