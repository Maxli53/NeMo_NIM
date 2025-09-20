# 🤖 Professional Multi-Agent Discussion System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A sophisticated, production-ready framework for orchestrating multi-agent discussions using GPT-OSS 20B MoE, RAG, and local embeddings. Features Claude Desktop integration via MCP, professional Streamlit dashboard, and complete offline operation capability.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The Multi-Agent Discussion System simulates interdisciplinary collaboration between domain experts (Physics, Biology, AI/ML, Chemistry) to solve complex problems. Each agent brings domain-specific knowledge via RAG (Retrieval-Augmented Generation), reasons collectively, and reaches consensus through voting mechanisms.

### Core Capabilities

- **🧠 Local LLM Support**: GPT-OSS 20B with Mixture-of-Experts (MoE) routing
- **📚 RAG Integration**: FAISS vector databases with local embeddings (Sentence Transformers)
- **🤝 Multi-Agent Orchestration**: Parallel async execution with consensus mechanisms
- **🖥️ Claude Desktop Integration**: MCP server for seamless integration
- **📊 Professional Dashboard**: Real-time Streamlit visualization
- **🔌 Multiple Interfaces**: CLI, Web UI, REST API, MCP
- **💾 Session Management**: Complete audit trail and export capabilities
- **🚀 Production Ready**: Logging, monitoring, error handling, and scaling

## ✨ Key Features

### 1. **Fully Offline Operation**
- Local GPT-OSS 20B MoE inference
- Sentence Transformers for embeddings
- No external API dependencies required

### 2. **Domain-Specific RAG**
- Each agent maintains separate FAISS vector database
- PDF processing with intelligent chunking
- Citation tracking and source attribution

### 3. **Advanced Consensus Mechanisms**
- Novelty and Feasibility voting
- Weighted consensus evaluation
- Configurable thresholds and criteria

### 4. **Professional Interfaces**
- **CLI**: Rich console with progress bars
- **Web UI**: Interactive Streamlit dashboard
- **API**: RESTful endpoints + WebSocket
- **MCP**: Claude Desktop integration

### 5. **Enterprise Features**
- Session persistence and replay
- Comprehensive logging
- Performance monitoring
- Horizontal scaling support

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Interfaces                      │
├──────────┬───────────┬──────────┬───────────┬──────────┤
│   CLI    │ Streamlit │   MCP    │  REST API │ WebSocket│
└──────────┴───────────┴──────────┴───────────┴──────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   Core Orchestration                     │
├──────────────────────────────────────────────────────────┤
│  • Discussion Moderator                                  │
│  • Session Manager                                       │
│  • Queue Management                                      │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                      Agent Layer                         │
├──────────┬──────────┬──────────┬──────────┬────────────┤
│ Physics  │ Biology  │   AI/ML  │Chemistry │ Consensus  │
│  Expert  │  Expert  │ Researcher│  Expert  │   Agent    │
└──────────┴──────────┴──────────┴──────────┴────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                    Knowledge Layer                       │
├──────────────────────────────────────────────────────────┤
│  • FAISS Vector Databases (per agent)                   │
│  • PDF Processing & Chunking                            │
│  • Local Embeddings (Sentence Transformers)             │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                     Model Layer                          │
├──────────────────────────────────────────────────────────┤
│  • GPT-OSS 20B MoE (Local)                              │
│  • Sentence Transformers (all-MiniLM-L6-v2)             │
│  • Optional: API fallback (Anthropic/OpenAI)            │
└─────────────────────────────────────────────────────────┘
```

### Project Structure

```
AI_agents/
├── main.py                      # Professional entry point (5 modes)
├── requirements.txt             # All dependencies
├── .env                        # Configuration (API keys optional)
├── src/
│   ├── config.py               # Enhanced configuration system
│   ├── agents/
│   │   ├── base.py            # Base agent with voting extraction
│   │   ├── expert.py          # Expert agent with local inference
│   │   └── consensus.py       # Consensus agent with weighted voting
│   ├── core/
│   │   ├── model_manager.py   # GPT-OSS 20B MoE manager
│   │   ├── vector_db.py       # Enhanced FAISS with persistence
│   │   ├── moderator.py       # Async discussion orchestration
│   │   └── session.py         # Professional session management
│   ├── utils/
│   │   ├── embeddings.py      # Local embedding manager
│   │   └── pdf_processor.py   # Enhanced PDF processing
│   ├── ui/
│   │   └── streamlit_app.py   # Professional dashboard
│   └── mcp_server.py          # Claude Desktop MCP server
├── data/
│   ├── pdfs/                  # Agent knowledge bases
│   └── indices/               # Persisted FAISS indices
├── sessions/                   # Session exports (JSON)
└── logs/                      # Application logs
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 32GB+ RAM (for GPT-OSS 20B)
- GPT-OSS 20B model files

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AI_agents.git
cd AI_agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run initial test
python main.py --mode test
```

### Quick Run

```bash
# Start Streamlit UI (recommended)
python main.py --mode ui

# Run CLI discussion
python main.py --mode cli --task "Design a quantum computer"

# Start MCP server for Claude Desktop
python main.py --mode mcp

# Launch REST API server
python main.py --mode api
```

## 📦 Installation

### Detailed Installation Steps

#### 1. **System Requirements**

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| CPU | 8 cores | 16+ cores | For parallel agent execution |
| RAM | 32 GB | 64 GB | For loading GPT-OSS 20B |
| GPU | RTX 3090 (24GB) | A100 (40GB) | Optional but recommended |
| Storage | 100 GB SSD | 500 GB NVMe | For model and indices |
| Python | 3.10 | 3.11 | Tested on both versions |

#### 2. **Environment Setup**

```bash
# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install -r requirements.txt

# Install optional performance enhancers
pip install accelerate bitsandbytes  # For quantization support
```

#### 3. **Model Installation**

```bash
# Download GPT-OSS 20B (using Hugging Face)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('gpt-oss-20b')"

# Or manually place model files in:
mkdir -p models/gpt-oss-20b
# Copy your model files here
```

#### 4. **Knowledge Base Setup**

```bash
# Create directory structure
mkdir -p data/pdfs data/indices sessions logs

# Add domain-specific PDFs
cp /path/to/physics.pdf data/pdfs/
cp /path/to/biology.pdf data/pdfs/
cp /path/to/ai_ml.pdf data/pdfs/
cp /path/to/chemistry.pdf data/pdfs/
```

## ⚙️ Configuration

### Environment Variables (.env)

```env
# Optional API Keys (for fallback mode)
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Model Configuration
MODEL_PROVIDER=gpt_oss          # Options: gpt_oss, anthropic, openai, local
MODEL_NAME=gpt-oss-20b
MODEL_DEVICE=auto               # Options: auto, cuda, cpu
MODEL_QUANTIZATION=8bit         # Options: none, 8bit, 4bit

# Embedding Configuration
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cuda           # Options: cuda, cpu

# Server Configuration
MCP_HOST=0.0.0.0
MCP_PORT=8000
STREAMLIT_PORT=8501

# Performance Settings
MAX_WORKERS=4
CACHE_EMBEDDINGS=true
BATCH_SIZE=32
```

### Advanced Configuration (src/config.py)

The system uses a comprehensive configuration system with dataclasses:

- **ModelConfig**: LLM settings, MoE parameters, quantization
- **EmbeddingConfig**: Embedding model, dimension, batch size
- **AgentConfig**: Individual agent settings, PDF paths
- **ConsensusConfig**: Voting thresholds, weights, rounds
- **MCPConfig**: Server host, port, CORS settings
- **RAGConfig**: Chunk size, overlap, retrieval parameters
- **UIConfig**: Streamlit theme, chart settings, export options
- **LoggingConfig**: Log levels, rotation, session logging

## 📖 Usage

### 1. **Command Line Interface (CLI)**

```bash
# Basic discussion with rich console output
python main.py --mode cli

# Specific task with parameters
python main.py --mode cli \
  --task "Design a bio-inspired quantum computer" \
  --rounds 5 \
  --knowledge

# Custom PDFs for each agent
python main.py --mode cli \
  --pdf physics_quantum.pdf \
  --pdf biology_neurons.pdf \
  --pdf ml_architectures.pdf \
  --pdf chemistry_materials.pdf
```

### 2. **Streamlit Dashboard**

```bash
# Start the dashboard
python main.py --mode ui

# Access at http://localhost:8501
```

**Features:**
- Real-time discussion visualization
- Agent messages with color coding
- Vote tracking with live charts
- Citation panel with expandable details
- Session statistics and metrics
- Export to JSON/CSV

### 3. **MCP Server (Claude Desktop)**

```bash
# Start MCP server
python main.py --mode mcp

# Server runs on http://localhost:8000
```

**Configure Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "multiAgentDiscussion": {
      "command": "python",
      "args": ["C:/path/to/AI_agents/main.py", "--mode", "mcp"]
    }
  }
}
```

**Use in Claude Desktop:**
```
@multiAgentDiscussion.run_discussion task="Design a sustainable city" max_rounds=5
@multiAgentDiscussion.get_session session_id="20240101_120000"
@multiAgentDiscussion.list_sessions
```

### 4. **REST API**

```bash
# Start API server
python main.py --mode api

# Server runs on http://localhost:8000
```

**Example API Calls:**

```bash
# Start discussion
curl -X POST http://localhost:8000/api/v1/discussion \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Design a quantum neural network",
    "max_rounds": 5,
    "use_knowledge_base": true
  }'

# Get session details
curl http://localhost:8000/api/v1/session/20240101_120000

# List all sessions
curl http://localhost:8000/api/v1/sessions

# Health check
curl http://localhost:8000/health
```

### 5. **WebSocket (Real-time Streaming)**

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/discussion');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`${data.agent}: ${data.message}`);
};

ws.send(JSON.stringify({
  action: "start_discussion",
  task: "Design a quantum computer",
  max_rounds: 5
}));
```

### 6. **Python SDK Usage**

```python
import asyncio
from src.core import MultiAgentSystem

async def main():
    # Initialize system
    system = MultiAgentSystem()
    await system.initialize()

    # Run discussion
    results = await system.run_discussion(
        task="Design a sustainable energy system",
        max_rounds=5,
        use_knowledge_base=True,
        pdf_paths=["energy.pdf", "biology.pdf"]
    )

    # Access results
    print(f"Consensus Reached: {results['consensus_reached']}")
    print(f"Average Novelty: {results['avg_novelty']:.1f}/10")
    print(f"Average Feasibility: {results['avg_feasibility']:.1f}/10")

    # Export session
    session_path = results['session'].export_session()
    print(f"Session saved to: {session_path}")

asyncio.run(main())
```

## 📊 Monitoring & Analytics

### Session Metrics

The system tracks comprehensive metrics:

- **Performance Metrics**
  - Token generation speed (tokens/sec)
  - Embedding generation time (ms)
  - Retrieval latency (ms)
  - Total processing time (sec)
  - Cache hit rates (%)

- **Quality Metrics**
  - Average novelty scores (0-10)
  - Average feasibility scores (0-10)
  - Consensus convergence rate
  - Citation coverage (%)
  - Agent participation balance

- **Resource Metrics**
  - GPU memory usage (GB)
  - CPU utilization (%)
  - Model cache efficiency
  - Queue depths
  - Active connections

### Logging Structure

```
logs/
├── app.log              # Main application log
├── agents.log           # Agent-specific activities
├── model.log           # Model inference details
├── embeddings.log      # Embedding generation
├── performance.log     # Performance metrics
├── mcp_server.log      # MCP server operations
└── sessions/
    └── 20240101/       # Daily session logs
        └── session_120000.log
```

## 🔧 Advanced Features

### 1. **Custom Domain Agents**

```python
from src.agents.expert import ExpertAgent
from src.core.vector_db import FAISSVectorDB

class QuantumExpert(ExpertAgent):
    def __init__(self):
        # Initialize with quantum-specific knowledge
        quantum_db = FAISSVectorDB()
        quantum_db.load("data/indices/quantum_db")

        super().__init__(
            name="QuantumExpert",
            domain="Quantum Computing and Information",
            vector_db=quantum_db,
            temperature=0.6  # Lower temperature for precision
        )

    async def preprocess_query(self, query: str) -> str:
        # Add quantum-specific context
        return f"From a quantum computing perspective: {query}"

# Register the agent
config.agents_config.append(
    AgentConfig(name="QuantumExpert", domain="Quantum Computing")
)
```

### 2. **Custom Consensus Mechanisms**

```python
from src.agents.consensus import ConsensusAgent

class WeightedConsensus(ConsensusAgent):
    def __init__(self, expertise_weights: Dict[str, float]):
        super().__init__()
        self.expertise_weights = expertise_weights

    def evaluate_consensus(self, votes: List[Dict]) -> Tuple[bool, float, float]:
        # Apply expertise-based weighting
        weighted_novelty = 0
        weighted_feasibility = 0
        total_weight = 0

        for vote in votes:
            agent_name = vote['agent']
            weight = self.expertise_weights.get(agent_name, 1.0)

            weighted_novelty += vote['novelty'] * weight
            weighted_feasibility += vote['feasibility'] * weight
            total_weight += weight

        avg_novelty = weighted_novelty / total_weight
        avg_feasibility = weighted_feasibility / total_weight

        consensus = (avg_novelty >= self.novelty_threshold and
                    avg_feasibility >= self.feasibility_threshold)

        return consensus, avg_novelty, avg_feasibility
```

### 3. **Dynamic Knowledge Base Updates**

```python
from src.core.vector_db import vector_db_manager
from src.utils.pdf_processor import PDFProcessor
from src.utils.embeddings import embedding_manager

async def update_agent_knowledge(agent_name: str, pdf_path: str):
    # Get agent's vector DB
    vector_db = vector_db_manager.get_or_create_db(agent_name)

    # Process new PDF
    processor = PDFProcessor()
    chunks = processor.chunk_pdf(pdf_path)

    # Generate embeddings
    texts = [chunk["text"] for chunk in chunks]
    sources = [chunk["source"] for chunk in chunks]
    result = embedding_manager.embed_batch(texts)

    # Update vector DB
    vector_db.add_batch(result.embeddings, texts, sources)

    # Save to disk
    vector_db.save(f"data/indices/{agent_name}_db")

    print(f"Updated {agent_name} with {len(chunks)} new chunks")
```

### 4. **Session Replay and Analysis**

```python
from src.core.session import SessionManager
import json

# Load previous session
session_manager = SessionManager()
session = session_manager.load_session("sessions/session_20240101_120000.json")

# Analyze voting patterns
novelty_scores = []
feasibility_scores = []

for entry in session.session_log:
    if entry.get('novelty_score'):
        novelty_scores.append(entry['novelty_score'])
    if entry.get('feasibility_score'):
        feasibility_scores.append(entry['feasibility_score'])

# Generate report
report = {
    "session_id": session.session_id,
    "task": session.task,
    "convergence_rate": calculate_convergence(novelty_scores, feasibility_scores),
    "agent_contributions": analyze_contributions(session.session_log),
    "citation_quality": assess_citations(session.session_log)
}

# Export enhanced report
with open(f"reports/{session.session_id}_analysis.json", "w") as f:
    json.dump(report, f, indent=2)
```

### 5. **Performance Optimization**

```python
# Enable 8-bit quantization for memory efficiency
from src.core.model_manager import model_manager

model_manager.enable_quantization("8bit")

# Use batch inference for multiple agents
async def batch_agent_inference(agents: List[ExpertAgent], prompts: List[str]):
    tasks = []
    for agent, prompt in zip(agents, prompts):
        tasks.append(agent.generate_async(prompt))

    responses = await asyncio.gather(*tasks)
    return responses

# Cache frequently used embeddings
embedding_manager.enable_cache(max_size=10000)

# Use GPU for FAISS if available
vector_db = FAISSVectorDB(use_gpu=True)
```

## 🚢 Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 8000 8501

# Run application
CMD ["python3", "main.py", "--mode", "api"]
```

```bash
# Build and run
docker build -t multi-agent-system .
docker run -p 8000:8000 -p 8501:8501 --gpus all multi-agent-system
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-server:
    build: .
    command: python main.py --mode mcp
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./sessions:/app/sessions
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  streamlit:
    build: .
    command: python main.py --mode ui
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./sessions:/app/sessions

  api:
    build: .
    command: python main.py --mode api
    ports:
      - "8080:8000"
    volumes:
      - ./data:/app/data
      - ./sessions:/app/sessions
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: multi-agent-system
  namespace: ai-agents
spec:
  replicas: 3
  selector:
    matchLabels:
      app: multi-agent
  template:
    metadata:
      labels:
        app: multi-agent
    spec:
      containers:
      - name: agent-system
        image: multi-agent:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "64Gi"
            cpu: "16"
          requests:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: models
          mountPath: /app/models
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: agent-data-pvc
      - name: models
        persistentVolumeClaim:
          claimName: model-storage-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: multi-agent-service
  namespace: ai-agents
spec:
  selector:
    app: multi-agent
  ports:
  - name: mcp
    port: 8000
    targetPort: 8000
  - name: streamlit
    port: 8501
    targetPort: 8501
  type: LoadBalancer
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. **Out of Memory (OOM) Errors**

**Problem:** Model fails to load with OOM error

**Solutions:**
```bash
# Enable 8-bit quantization
MODEL_QUANTIZATION=8bit python main.py

# Use CPU offloading
MODEL_DEVICE=cpu python main.py

# Reduce batch size
BATCH_SIZE=8 python main.py
```

#### 2. **Slow Inference Speed**

**Problem:** Agent responses are very slow

**Solutions:**
```bash
# Check GPU availability
nvidia-smi

# Enable MoE expert caching
python -c "from src.core.model_manager import model_manager; model_manager.enable_expert_cache()"

# Use smaller embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2 python main.py
```

#### 3. **FAISS Index Errors**

**Problem:** Vector database search fails

**Solutions:**
```python
# Rebuild indices
from src.core.vector_db import vector_db_manager

vector_db_manager.clear_all()
vector_db_manager.rebuild_indices()

# Verify embedding dimensions
print(f"Expected: {config.embedding_config.dimension}")
print(f"Actual: {embedding_manager.dimension}")
```

#### 4. **MCP Connection Issues**

**Problem:** Claude Desktop can't connect to MCP server

**Solutions:**
1. Check server is running: `curl http://localhost:8000/health`
2. Verify Claude Desktop config path is absolute
3. Check firewall settings
4. Review MCP server logs: `tail -f logs/mcp_server.log`

#### 5. **PDF Processing Failures**

**Problem:** PDFs fail to process or chunk

**Solutions:**
```python
# Debug PDF processing
from src.utils.pdf_processor import PDFProcessor

processor = PDFProcessor(chunk_size=300, overlap=50)
try:
    chunks = processor.chunk_pdf("path/to/pdf")
    print(f"Successfully processed {len(chunks)} chunks")
except Exception as e:
    print(f"Error: {e}")
    # Try with different parameters
    processor = PDFProcessor(chunk_size=200, overlap=25)
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/AI_agents.git
cd AI_agents

# Create development environment
python -m venv venv-dev
source venv-dev/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .  # Install package in editable mode

# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/

# Lint code
pylint src/
mypy src/

# Generate documentation
sphinx-build docs/ docs/_build/
```

### Code Standards

- **Style**: Follow PEP 8, use Black formatter
- **Types**: Add type hints to all functions
- **Docs**: Write docstrings for all public functions
- **Tests**: Maintain >80% test coverage
- **Commits**: Use conventional commits format

### Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'feat: add amazing feature'`
4. Push branch: `git push origin feature/amazing-feature`
5. Open Pull Request with detailed description

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GPT-OSS Community** for the MoE model architecture
- **Hugging Face** for Transformers library
- **Meta AI** for FAISS vector database
- **Streamlit** for the dashboard framework
- **Anthropic** for Claude Desktop MCP protocol
- **Sentence Transformers** for local embeddings

## 📚 Additional Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Detailed system architecture
- [API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) - Complete API reference
- [CONFIGURATION.md](docs/CONFIGURATION.md) - Configuration guide
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Deployment strategies
- [DEVELOPMENT.md](docs/DEVELOPMENT.md) - Development guidelines
- [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - Extended troubleshooting

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/AI_agents/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/AI_agents/discussions)
- **Documentation**: [Read the Docs](https://ai-agents.readthedocs.io)
- **Email**: support@ai-agents.dev

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/AI_agents&type=Date)](https://star-history.com/#yourusername/AI_agents&Date)

---

**Version**: 1.0.0 | **Last Updated**: January 2024 | **Status**: Production Ready