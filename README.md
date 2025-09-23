# AI Agents - Multi-Agent Discussion System with MoE Backend

## 🎯 Overview

A professional multi-agent discussion system that leverages multiple AI models including GPT-OSS-20B through an optimized Mixture of Experts (MoE) implementation.

## 🏗️ Architecture

This project consists of two integrated systems:

### 1. Multi-Agent Discussion System (Main Application)
- **Entry Point**: `main.py`
- **Purpose**: Orchestrate discussions between multiple AI agents
- **Features**:
  - Multiple agent types (Expert, Consensus)
  - RAG (Retrieval Augmented Generation)
  - PDF document processing
  - Vector database for knowledge storage
  - Streamlit UI and FastAPI endpoints
  - Support for multiple model providers

### 2. MoE Backend (GPT-OSS-20B Support)
- **Location**: `src/moe/`
- **Purpose**: High-performance inference for GPT-OSS-20B model
- **Features**:
  - Top-k expert selection (4 out of 32 experts)
  - FP16 precision with SDPA/Flash Attention
  - Async expert loading and tiered caching
  - Safety framework with feature flags
  - **Performance**: 29.1 tokens/sec with 7.3GB VRAM

## 📁 Project Structure

```
AI_agents/
├── main.py                 # Entry point for agent system
├── src/
│   ├── agents/            # Multi-agent implementations
│   │   ├── expert.py      # Expert agent
│   │   ├── consensus.py  # Consensus agent
│   │   └── integrated_multi_agent_gptoss.py
│   ├── core/              # Core system components
│   │   ├── moderator.py  # Discussion moderator
│   │   ├── session.py    # Session management
│   │   ├── vector_db.py  # Vector database
│   │   └── model_manager.py
│   ├── moe/               # MoE implementation for GPT-OSS
│   │   ├── native_moe_loader_v2.py  # Main MoE loader
│   │   ├── expert_cache.py          # LRU caching
│   │   ├── async_expert_loader.py   # Async loading
│   │   ├── tiered_cache.py          # Tiered caching
│   │   └── optimization_safety/     # Safety framework
│   ├── api/               # API endpoints
│   │   └── server.py      # FastAPI server
│   ├── ui/                # User interfaces
│   │   └── streamlit_app.py
│   └── utils/             # Utilities
├── tests/                 # Test suites
├── scripts/               # Utility scripts
├── configs/               # Configuration files
└── docs/                  # Documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10-3.12
- CUDA 12.8+ (for GPU acceleration)
- 24GB+ VRAM (for full GPT-OSS-20B)
- WSL2 (Windows) or Linux

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI_agents.git
cd AI_agents
```

2. Create virtual environment:
```bash
python -m venv venv_wsl
source venv_wsl/bin/activate  # Linux/WSL
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

### Running the System

#### Option 1: Multi-Agent Discussion (Main)
```bash
python main.py --model gpt-oss --topic "Your discussion topic"
```

#### Option 2: Streamlit UI
```bash
streamlit run src/ui/streamlit_app.py
```

#### Option 3: API Server
```bash
python -m src.api.server
# API available at http://localhost:8000
```

#### Option 4: Direct MoE Inference
```bash
python -c "from src.moe.native_moe_loader_v2 import MoEModelLoader; loader = MoEModelLoader('gpt-oss-20b/original'); model = loader.create_model_fp16(top_k=4)"
```

## 🎯 Supported Models

The system supports multiple model providers:

| Provider | Model | Backend | Status |
|----------|-------|---------|--------|
| GPT-OSS | gpt-oss-20b | MoE Implementation | ✅ Production Ready |
| Anthropic | Claude 3 | API | ✅ Ready |
| OpenAI | GPT-4 | API | ✅ Ready |
| Local | Various | Transformers | ✅ Ready |

## 📊 Performance

### GPT-OSS-20B with MoE Optimizations
- **Throughput**: 29.1 tokens/second
- **Memory**: 7.3GB VRAM (vs 40GB+ baseline)
- **First Token Latency**: 30ms
- **Configuration**: FP16 + SDPA + Top-k=4

### Optimization Status
| Optimization | Status | Impact |
|--------------|--------|--------|
| FP16 Precision | ✅ Enabled | Baseline |
| SDPA/Flash Attention | ✅ Enabled | +15% speed |
| Top-k Expert Selection | ✅ Enabled (k=4) | -87% memory |
| Async Loading | ✅ Enabled | Faster startup |
| torch.compile | ❌ Disabled | -88% speed (regression) |
| INT8 Quantization | ❌ Disabled | -80% speed (issues) |

## 🛡️ Safety Features

- **Feature Flags**: Centralized control for all optimizations
- **Health Monitoring**: Real-time performance tracking
- **Automatic Rollback**: Reverts on performance degradation
- **Error Handling**: Comprehensive error management
- **Logging**: Structured logging throughout

## 📖 Documentation

- [Technical Architecture](docs/TECHNICAL.md) - System design details
- [Performance Guide](docs/PERFORMANCE.md) - Optimization benchmarks
- [Operations Manual](docs/OPERATIONS.md) - Deployment guide
- [Development Guide](docs/DEVELOPMENT.md) - Contributing guidelines
- [Best Practices](docs/BEST_PRACTICES.md) - Code standards

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Unit tests only
pytest tests/test_unit.py -v

# Performance benchmarks
pytest tests/test_performance.py -v

# Functional tests
pytest tests/test_functional.py -v
```

## 🤝 Contributing

1. Follow guidelines in [CLAUDE.md](CLAUDE.md)
2. Ensure all tests pass
3. Update documentation
4. Submit PR with clear description

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- OpenAI for GPT-OSS-20B model
- PyTorch team for framework
- NVIDIA for CUDA and Flash Attention

## 📞 Support

- Issues: [GitHub Issues](https://github.com/yourusername/AI_agents/issues)
- Documentation: [docs/](docs/)
- Performance Issues: Check [PERFORMANCE.md](docs/PERFORMANCE.md)

---

**Note**: This is an integrated system combining multi-agent orchestration with high-performance MoE inference for GPT-OSS-20B. Both components work together to provide a complete AI discussion platform.