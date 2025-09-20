# 🔧 Configuration Guide

## Table of Contents
- [Overview](#overview)
- [Configuration Files](#configuration-files)
- [Environment Variables](#environment-variables)
- [Model Configuration](#model-configuration)
- [Agent Configuration](#agent-configuration)
- [Embedding Configuration](#embedding-configuration)
- [Vector Database Configuration](#vector-database-configuration)
- [MCP Server Configuration](#mcp-server-configuration)
- [Logging Configuration](#logging-configuration)
- [Performance Tuning](#performance-tuning)
- [Advanced Configuration](#advanced-configuration)
- [Configuration Examples](#configuration-examples)
- [Troubleshooting](#troubleshooting)

## Overview

The Multi-Agent Discussion System provides flexible configuration options through:
- Environment variables
- Configuration files (`config.yaml`, `.env`)
- Command-line arguments
- Runtime configuration via API

Configuration priority (highest to lowest):
1. Command-line arguments
2. Environment variables
3. Configuration files
4. Default values

## Configuration Files

### Main Configuration File (`config.yaml`)

```yaml
# config.yaml - Main configuration file
version: "1.0.0"

# Model Configuration
model:
  provider: "local"  # local, openai, anthropic
  model_name: "gpt-oss-20b"  # Model identifier
  
  # Local Model Settings (GPT-OSS 20B MoE)
  local:
    model_path: "./models/gpt-oss-20b"  # Path to model files
    device: "auto"  # cuda, cpu, auto
    quantization: "8bit"  # none, 8bit, 4bit
    max_memory: "24GB"  # Maximum GPU memory
    offload_folder: "./offload"  # CPU offload directory
    
    # MoE Configuration
    moe:
      total_experts: 128  # Total number of experts
      active_experts: 32  # Active experts per token
      expert_capacity: 256  # Tokens per expert
      load_balancing: true  # Enable load balancing
      router_temperature: 0.1  # Router softmax temperature
    
    # Generation Parameters
    generation:
      max_length: 2048
      temperature: 0.7
      top_p: 0.9
      top_k: 50
      repetition_penalty: 1.1
      do_sample: true
  
  # API Model Settings
  api:
    api_key: "${OPENAI_API_KEY}"  # From environment
    organization: "${OPENAI_ORG}"  # Optional
    base_url: null  # Custom API endpoint
    timeout: 60  # Request timeout in seconds
    max_retries: 3
    retry_delay: 1.0

# Embedding Configuration
embedding:
  provider: "local"  # local, openai
  model_name: "all-MiniLM-L6-v2"  # Sentence transformer model
  
  local:
    model_path: null  # Auto-download if null
    device: "auto"  # cuda, cpu, auto
    batch_size: 32
    normalize: true  # Normalize embeddings
    dimension: 384  # Embedding dimension
    max_sequence_length: 512
    
  cache:
    enabled: true
    ttl: 3600  # Cache TTL in seconds
    max_size: 10000  # Maximum cached embeddings
    path: "./cache/embeddings"

# Vector Database Configuration
vector_db:
  backend: "faiss"  # faiss, pinecone, weaviate, qdrant
  
  faiss:
    index_type: "IVF"  # Flat, IVF, HNSW
    nlist: 100  # Number of clusters for IVF
    nprobe: 10  # Number of clusters to search
    use_gpu: false  # GPU acceleration
    normalize: true  # L2 normalize vectors
    
    persistence:
      enabled: true
      path: "./data/indices"
      autosave_interval: 300  # Seconds
      compression: true
  
  retrieval:
    top_k: 5  # Number of results
    score_threshold: 0.7  # Minimum similarity
    rerank: true  # Enable reranking
    diversity: 0.3  # Result diversity (0-1)

# Agent Configuration
agents:
  # Expert Agents
  experts:
    - name: "PhysicsExpert"
      domain: "quantum mechanics, relativity, particle physics"
      temperature: 0.7
      max_tokens: 1000
      knowledge_base:
        enabled: true
        pdf_path: "./data/physics/textbook.pdf"
        index_name: "physics_index"
      personality:
        style: "analytical"
        verbosity: "moderate"
        citation_style: "academic"
    
    - name: "BiologyExpert"
      domain: "molecular biology, genetics, neuroscience"
      temperature: 0.6
      max_tokens: 1000
      knowledge_base:
        enabled: true
        pdf_path: "./data/biology/textbook.pdf"
        index_name: "biology_index"
      personality:
        style: "empirical"
        verbosity: "detailed"
        citation_style: "scientific"
    
    - name: "AIResearcher"
      domain: "machine learning, deep learning, AGI"
      temperature: 0.8
      max_tokens: 1000
      knowledge_base:
        enabled: true
        pdf_path: "./data/ai/papers.pdf"
        index_name: "ai_index"
      personality:
        style: "innovative"
        verbosity: "concise"
        citation_style: "technical"
    
    - name: "ChemistryExpert"
      domain: "organic chemistry, materials science"
      temperature: 0.65
      max_tokens: 1000
      knowledge_base:
        enabled: true
        pdf_path: "./data/chemistry/handbook.pdf"
        index_name: "chemistry_index"
      personality:
        style: "methodical"
        verbosity: "moderate"
        citation_style: "standard"
  
  # Consensus Agent
  consensus:
    name: "ConsensusAgent"
    temperature: 0.3  # Lower for consistency
    max_tokens: 1500
    thresholds:
      novelty: 7.0  # Minimum novelty score
      feasibility: 6.0  # Minimum feasibility score
      confidence: 0.8  # Consensus confidence
    evaluation:
      weighted_voting: true
      expert_weights:
        PhysicsExpert: 1.0
        BiologyExpert: 1.0
        AIResearcher: 1.2  # Higher weight for AI
        ChemistryExpert: 1.0

# Discussion Configuration
discussion:
  max_rounds: 10
  min_rounds: 3
  parallel_responses: true  # Agents respond in parallel
  timeout_per_round: 120  # Seconds
  
  moderation:
    style: "socratic"  # socratic, debate, collaborative
    interventions: true  # Moderator can intervene
    summarize_rounds: true  # Summarize after each round
  
  voting:
    enabled: true
    anonymous: false
    criteria:
      - "novelty"
      - "feasibility"
      - "scientific_rigor"
      - "interdisciplinary_value"

# MCP Server Configuration
mcp:
  enabled: true
  host: "127.0.0.1"
  port: 8000
  
  api:
    prefix: "/api/v1"
    docs_url: "/docs"
    redoc_url: "/redoc"
  
  cors:
    origins:
      - "http://localhost:*"
      - "http://127.0.0.1:*"
      - "claude://desktop"
    allow_credentials: true
    allow_methods: ["*"]
    allow_headers: ["*"]
  
  websocket:
    enabled: true
    path: "/ws"
    heartbeat_interval: 30
    max_connections: 100
  
  security:
    api_key_required: false  # Set to true in production
    api_key_header: "X-API-Key"
    rate_limiting:
      enabled: true
      requests_per_minute: 60
      burst: 10

# Logging Configuration
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  handlers:
    console:
      enabled: true
      rich_format: true  # Use rich formatting
      traceback: true  # Show tracebacks
    
    file:
      enabled: true
      path: "./logs/discussion.log"
      rotation: "daily"  # daily, size, time
      retention: 7  # Days to keep
      compression: true
    
    remote:
      enabled: false
      endpoint: "https://logging.example.com"
      api_key: "${LOGGING_API_KEY}"
  
  modules:
    # Set specific log levels per module
    "src.agents": "DEBUG"
    "src.core": "INFO"
    "src.utils": "WARNING"
    "transformers": "ERROR"  # Reduce transformer noise

# Performance Configuration
performance:
  # Parallelization
  parallel:
    max_workers: 4  # Thread pool size
    async_batch_size: 10  # Async batch processing
  
  # Caching
  cache:
    embeddings: true
    model_outputs: true
    retrieval_results: true
    ttl: 3600  # Default TTL in seconds
  
  # Memory Management
  memory:
    garbage_collection_interval: 300  # Seconds
    max_session_memory: "4GB"
    clear_gpu_cache: true
  
  # Optimization
  optimization:
    compile_model: false  # PyTorch 2.0 compilation
    mixed_precision: true  # FP16/BF16 training
    gradient_checkpointing: false
    cpu_offload: true  # Offload to CPU when needed

# Session Configuration
session:
  storage:
    backend: "filesystem"  # filesystem, database, redis
    path: "./data/sessions"
    format: "json"  # json, pickle, parquet
  
  management:
    auto_save: true
    save_interval: 60  # Seconds
    max_sessions: 1000
    cleanup_old_sessions: true
    retention_days: 30
  
  export:
    formats: ["json", "markdown", "pdf"]
    include_metadata: true
    include_citations: true
    compress: false

# UI Configuration
ui:
  streamlit:
    port: 8501
    theme: "dark"  # light, dark, auto
    wide_mode: true
    show_citations: true
    auto_refresh: true
    refresh_interval: 1  # Seconds
  
  components:
    chat_height: 600  # Pixels
    sidebar_width: 300
    show_metrics: true
    show_timeline: true
    show_voting_charts: true

# Development Configuration
development:
  debug: false
  hot_reload: true
  profiling: false
  test_mode: false
  
  mock_data:
    enabled: false
    response_delay: 0.5  # Simulate processing time
  
  monitoring:
    prometheus: false
    metrics_port: 9090
    health_check_path: "/health"
```

### Environment Variables (`.env`)

```bash
# .env file - Environment-specific configuration

# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HUGGINGFACE_TOKEN=hf_...

# Model Paths
MODEL_PATH=/path/to/gpt-oss-20b
EMBEDDING_MODEL_PATH=/path/to/embeddings

# Database
VECTOR_DB_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://user:pass@localhost/db

# Cloud Storage
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-west-2
S3_BUCKET=model-artifacts

# Monitoring
SENTRY_DSN=https://...@sentry.io/...
DATADOG_API_KEY=...

# Feature Flags
ENABLE_GPU=true
ENABLE_QUANTIZATION=true
ENABLE_CACHING=true
ENABLE_MONITORING=false

# Performance
MAX_WORKERS=4
BATCH_SIZE=32
CUDA_VISIBLE_DEVICES=0,1

# Deployment
ENVIRONMENT=development  # development, staging, production
LOG_LEVEL=INFO
DEBUG=false
```

## Environment Variables

### Required Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MODEL_PATH` | Path to GPT-OSS 20B model | `./models/gpt-oss-20b` | No |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` | No |
| `CUDA_VISIBLE_DEVICES` | GPU devices to use | `0` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |

### Optional API Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | None | If using OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic API key | None | If using Claude |
| `HUGGINGFACE_TOKEN` | HuggingFace token | None | For private models |

## Model Configuration

### GPT-OSS 20B MoE Configuration

```python
# src/config.py - Programmatic configuration

from dataclasses import dataclass
from typing import Optional

@dataclass
class MoEConfig:
    """Mixture of Experts configuration"""
    total_experts: int = 128
    active_experts: int = 32
    expert_capacity: int = 256
    load_balancing: bool = True
    router_temperature: float = 0.1
    router_noise: float = 0.01
    expert_dropout: float = 0.0
    
    def validate(self):
        assert self.active_experts <= self.total_experts
        assert self.router_temperature > 0
        assert 0 <= self.expert_dropout < 1

@dataclass
class QuantizationConfig:
    """Model quantization configuration"""
    enabled: bool = True
    bits: int = 8  # 4 or 8
    group_size: int = 128
    damp_percent: float = 0.01
    desc_act: bool = True
    sym: bool = True
    true_sequential: bool = True
```

### Model Loading Optimization

```python
# Optimized model loading configuration

model_config = {
    "device_map": "auto",  # Automatic device placement
    "max_memory": {
        0: "20GB",  # GPU 0
        1: "20GB",  # GPU 1
        "cpu": "30GB"  # CPU offload
    },
    "quantization_config": BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_use_double_quant=True
    ),
    "torch_dtype": torch.float16,
    "low_cpu_mem_usage": True
}
```

## Agent Configuration

### Custom Agent Creation

```python
# Custom agent configuration

custom_agent = {
    "name": "EconomicsExpert",
    "domain": "macroeconomics, financial markets, game theory",
    "model_config": {
        "temperature": 0.75,
        "max_tokens": 1200,
        "top_p": 0.95,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.1
    },
    "knowledge_base": {
        "sources": [
            "./data/economics/textbook.pdf",
            "./data/economics/papers/*.pdf"
        ],
        "chunk_size": 512,
        "chunk_overlap": 50,
        "embedding_batch_size": 16
    },
    "behavior": {
        "response_style": "analytical",
        "use_citations": True,
        "min_citations": 2,
        "max_citations": 5,
        "voting_weight": 1.1
    }
}
```

### Agent Personality Profiles

```yaml
# Agent personality configuration

personalities:
  analytical:
    style: "logical and systematic"
    language: "formal"
    reasoning: "deductive"
    evidence_preference: "quantitative"
    
  creative:
    style: "innovative and exploratory"
    language: "engaging"
    reasoning: "associative"
    evidence_preference: "diverse"
    
  empirical:
    style: "evidence-based"
    language: "precise"
    reasoning: "inductive"
    evidence_preference: "experimental"
```

## Embedding Configuration

### Sentence Transformer Models

```yaml
# Available embedding models

embedding_models:
  fast:
    model: "all-MiniLM-L6-v2"
    dimension: 384
    max_length: 256
    speed: "fast"
    quality: "good"
    
  balanced:
    model: "all-mpnet-base-v2"
    dimension: 768
    max_length: 384
    speed: "medium"
    quality: "better"
    
  accurate:
    model: "all-roberta-large-v1"
    dimension: 1024
    max_length: 512
    speed: "slow"
    quality: "best"
```

### Embedding Cache Configuration

```python
# Embedding cache settings

cache_config = {
    "backend": "redis",  # memory, redis, disk
    "redis": {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": None,
        "ttl": 3600,
        "max_connections": 10
    },
    "memory": {
        "max_size": 10000,
        "ttl": 3600,
        "eviction_policy": "lru"  # lru, lfu, fifo
    },
    "disk": {
        "path": "./cache/embeddings",
        "max_size_gb": 10,
        "compression": "gzip"
    }
}
```

## Vector Database Configuration

### FAISS Index Types

```python
# FAISS index configuration

index_configs = {
    "flat": {
        "type": "IndexFlatL2",
        "description": "Exact search, no training needed",
        "use_case": "Small datasets (<10k vectors)"
    },
    "ivf": {
        "type": "IndexIVFFlat",
        "nlist": 100,  # Number of clusters
        "nprobe": 10,  # Clusters to search
        "description": "Approximate search with clustering",
        "use_case": "Medium datasets (10k-1M vectors)"
    },
    "hnsw": {
        "type": "IndexHNSWFlat",
        "M": 32,  # Number of connections
        "ef_construction": 200,  # Construction time accuracy
        "ef_search": 100,  # Search time accuracy
        "description": "Graph-based approximate search",
        "use_case": "Large datasets with high recall needs"
    },
    "ivf_pq": {
        "type": "IndexIVFPQ",
        "nlist": 100,
        "m": 8,  # Number of subquantizers
        "bits": 8,  # Bits per subquantizer
        "description": "Compressed approximate search",
        "use_case": "Very large datasets with memory constraints"
    }
}
```

### Multi-Index Configuration

```yaml
# Multiple vector databases for different domains

vector_databases:
  physics:
    index_type: "hnsw"
    dimension: 768
    metric: "cosine"  # cosine, l2, ip
    gpu: false
    
  biology:
    index_type: "ivf"
    dimension: 768
    metric: "l2"
    gpu: false
    
  chemistry:
    index_type: "flat"
    dimension: 384
    metric: "cosine"
    gpu: false
```

## MCP Server Configuration

### Tool Registration

```python
# MCP tool configuration

mcp_tools = {
    "run_discussion": {
        "description": "Run multi-agent discussion",
        "parameters": {
            "task": {"type": "string", "required": True},
            "max_rounds": {"type": "integer", "default": 5},
            "agents": {"type": "array", "default": None}
        },
        "timeout": 300,
        "retries": 3
    },
    "get_session": {
        "description": "Get session details",
        "parameters": {
            "session_id": {"type": "string", "required": True}
        },
        "timeout": 30,
        "cache_ttl": 60
    }
}
```

### WebSocket Configuration

```python
# WebSocket settings

websocket_config = {
    "max_connections": 100,
    "connection_timeout": 60,
    "heartbeat_interval": 30,
    "message_queue_size": 1000,
    "compression": "deflate",
    "max_message_size": 1048576  # 1MB
}
```

## Logging Configuration

### Structured Logging

```python
# Structured logging configuration

import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
```

### Log Rotation

```python
# Log rotation configuration

from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

log_handlers = {
    "size_rotation": RotatingFileHandler(
        filename="app.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    ),
    "time_rotation": TimedRotatingFileHandler(
        filename="app.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8"
    )
}
```

## Performance Tuning

### GPU Optimization

```python
# GPU performance settings

import torch

# Enable TF32 for Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Optimize cudNN
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Memory management
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.9)
```

### CPU Optimization

```python
# CPU performance settings

import os

# OpenMP settings
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

# PyTorch CPU settings
torch.set_num_threads(8)
torch.set_num_interop_threads(4)
```

### Batch Processing

```yaml
# Batch processing configuration

batching:
  embedding:
    batch_size: 32
    max_batch_wait: 0.1  # seconds
    dynamic_batching: true
    
  inference:
    batch_size: 8
    padding: "max_length"
    truncation: true
    
  retrieval:
    batch_size: 100
    parallel_queries: 4
```

## Advanced Configuration

### Multi-GPU Setup

```python
# Multi-GPU configuration

multi_gpu_config = {
    "strategy": "ddp",  # ddp, dp, horovod
    "devices": [0, 1, 2, 3],
    "master_port": 29500,
    "find_unused_parameters": False,
    "gradient_as_bucket_view": True,
    "static_graph": True
}

# Device placement
device_map = {
    "embed_tokens": 0,
    "layers.0-15": 0,
    "layers.16-31": 1,
    "layers.32-47": 2,
    "lm_head": 3
}
```

### Distributed Configuration

```yaml
# Distributed system configuration

distributed:
  enabled: true
  backend: "nccl"  # nccl, gloo, mpi
  
  cluster:
    nodes:
      - host: "node1.example.com"
        port: 29500
        gpus: [0, 1]
      - host: "node2.example.com"
        port: 29500
        gpus: [0, 1]
    
  communication:
    compression: true
    compression_algorithm: "powersgd"
    bucket_cap_mb: 25
    
  synchronization:
    gradient_sync: true
    sync_batch_norm: true
    broadcast_buffers: true
```

### Security Configuration

```yaml
# Security settings

security:
  authentication:
    enabled: true
    type: "jwt"  # jwt, oauth2, api_key
    
    jwt:
      secret_key: "${JWT_SECRET}"
      algorithm: "HS256"
      expiration: 3600
    
    oauth2:
      provider: "auth0"
      domain: "example.auth0.com"
      client_id: "${OAUTH_CLIENT_ID}"
      client_secret: "${OAUTH_CLIENT_SECRET}"
  
  encryption:
    data_at_rest: true
    data_in_transit: true
    algorithm: "AES-256-GCM"
    
  rate_limiting:
    enabled: true
    strategy: "sliding_window"  # fixed_window, sliding_window
    limits:
      - endpoint: "/api/v1/discussion"
        requests: 10
        period: 60
      - endpoint: "/api/v1/embed"
        requests: 100
        period: 60
  
  input_validation:
    max_task_length: 10000
    max_rounds: 20
    allowed_file_types: [".pdf", ".txt", ".md"]
    max_file_size_mb: 50
```

## Configuration Examples

### Minimal Configuration

```yaml
# Minimal config for quick start

model:
  provider: "local"
  model_name: "gpt-oss-20b"

embedding:
  provider: "local"
  model_name: "all-MiniLM-L6-v2"

agents:
  experts:
    - name: "Expert1"
      domain: "general"
    - name: "Expert2"
      domain: "general"

mcp:
  enabled: true
  port: 8000
```

### Production Configuration

```yaml
# Production-ready configuration

model:
  provider: "local"
  model_name: "gpt-oss-20b"
  local:
    quantization: "8bit"
    device: "cuda"
    max_memory: "40GB"

embedding:
  provider: "local"
  model_name: "all-mpnet-base-v2"
  cache:
    enabled: true
    backend: "redis"

vector_db:
  faiss:
    index_type: "ivf"
    use_gpu: true
    persistence:
      enabled: true
      autosave_interval: 60

security:
  authentication:
    enabled: true
    type: "api_key"
  rate_limiting:
    enabled: true

logging:
  level: "INFO"
  handlers:
    remote:
      enabled: true

performance:
  optimization:
    mixed_precision: true
    compile_model: true
```

### Development Configuration

```yaml
# Development configuration with debugging

model:
  provider: "local"
  model_name: "gpt-oss-20b"
  local:
    quantization: "4bit"  # Faster loading
    device: "cpu"  # For debugging

development:
  debug: true
  hot_reload: true
  profiling: true
  mock_data:
    enabled: true

logging:
  level: "DEBUG"
  handlers:
    console:
      rich_format: true
      traceback: true
```

## Troubleshooting

### Common Configuration Issues

1. **Model Loading Errors**
   ```yaml
   # Fix: Ensure model path is correct
   model:
     local:
       model_path: "/absolute/path/to/model"
   ```

2. **Memory Issues**
   ```yaml
   # Fix: Reduce batch size and enable CPU offload
   performance:
     memory:
       max_session_memory: "2GB"
       cpu_offload: true
   ```

3. **GPU Not Detected**
   ```bash
   # Fix: Set CUDA_VISIBLE_DEVICES
   export CUDA_VISIBLE_DEVICES=0
   ```

4. **Embedding Cache Misses**
   ```yaml
   # Fix: Increase cache size and TTL
   embedding:
     cache:
       max_size: 50000
       ttl: 7200
   ```

### Configuration Validation

```python
# Validate configuration on startup

from src.config import config

try:
    config.validate()
    print("Configuration valid")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    sys.exit(1)
```

### Environment-Specific Overrides

```python
# Override configuration based on environment

import os

env = os.getenv("ENVIRONMENT", "development")

if env == "production":
    config.logging.level = "WARNING"
    config.mcp.security.api_key_required = True
    config.performance.cache.enabled = True
elif env == "development":
    config.logging.level = "DEBUG"
    config.development.debug = True
    config.development.hot_reload = True
```

---

For more configuration examples and best practices, see the [examples/configs](../examples/configs) directory.