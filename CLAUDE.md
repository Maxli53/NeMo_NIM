AI Expert Multi-Agent Discussion
Project Goals

The AI Expert Multi-Agent Discussion project aims to explore and implement a framework where multiple AI agents, each specializing in a specific domain, collaboratively reason, debate, and synthesize knowledge to solve complex interdisciplinary tasks. The project combines domain-specific retrieval, conceptual reasoning, voting, and consensus mechanisms with a real-time visualization dashboard.

Primary Goals

Simulate Interdisciplinary Collaboration

Emulate a team of experts (Physics, Biology, AI/ML, Chemistry) discussing a complex problem.

Enable agents to bring their domain-specific knowledge to the conversation.

Integrate Domain Knowledge

Retrieve relevant material from books, research papers, or other sources stored in FAISS vector databases.

Encourage agents to ground reasoning in actual sources while allowing conceptual reasoning if no material is available.

Enable Conceptual Reasoning

Allow agents to reason logically even without citations, ensuring discussion continues smoothly.

Implement Voting and Consensus Mechanisms

Each agent provides Novelty and Feasibility votes for each contribution.

The ConsensusAgent evaluates these votes to decide whether discussion has reached consensus.

Provide Transparency and Traceability

Citations are displayed in a dedicated panel, showing which sources influenced each agent’s reasoning.

Messages without relevant sources are explicitly noted, preserving honesty.

Visualize Interactions in Real-Time

Streamlit dashboard shows conversation threads per agent, votes over time, citations, and a timeline of contributions.

Support Incremental Experimentation

Stage testing from empty databases (conceptual reasoning only) to full domain-specific corpora.

Evaluate how adding knowledge sources affects discussion quality, citations, and consensus.

Secondary Goals

Encourage novel idea generation by combining multiple disciplines.

Enable reproducibility and analysis via session logs.

Create a scalable framework for future experiments, including adding more agents, domains, or live data feeds.

Expected Outcomes

A working multi-agent discussion system capable of coherent interdisciplinary reasoning.

Quantitative evaluation through Novelty and Feasibility votes.

Clear demonstration of the effect of domain-specific knowledge on reasoning quality.

A modular, extensible platform for research into collaborative AI problem-solving.

## Installation & Usage Examples

### Quick Installation

```bash
# Clone repository
git clone https://github.com/yourorg/ai-multi-agent.git
cd ai-multi-agent

# Install dependencies
pip install -r requirements.txt

# Download models (for local mode)
python scripts/download_models.py
```

### Configuration

The system now supports both local (GPT-OSS 20B MoE) and API modes:

```python
# Local mode (default) - no API keys required
MODEL_PROVIDER=local
MODEL_PATH=./models/gpt-oss-20b

# API mode (optional)
MODEL_PROVIDER=openai
OPENAI_API_KEY=your_key_here
```

### Usage Examples

#### 1. Command Line Interface (CLI)
```bash
# Run interactive CLI
python main.py --mode cli

# Run with specific task
python main.py --mode cli --task "Design a quantum-biological computer" --rounds 10

# Run with PDF knowledge bases
python main.py --mode cli --pdf data/physics.pdf --pdf data/biology.pdf
```

#### 2. Streamlit Dashboard (UI)
```bash
# Start the web dashboard
python main.py --mode ui

# Access at http://localhost:8501
```

#### 3. MCP Server (Claude Desktop Integration)
```bash
# Start MCP server
python main.py --mode mcp

# Configure Claude Desktop to connect to http://localhost:8000
```

#### 4. REST API Server
```bash
# Start API server
python main.py --mode api

# Make API calls
curl -X POST http://localhost:8000/api/v1/discussion \
  -H "Content-Type: application/json" \
  -d '{"task": "Design a novel energy storage system", "max_rounds": 5}'
```

#### 5. Python SDK Usage
```python
from src.agents.expert import ExpertAgent
from src.agents.consensus import ConsensusAgent
from src.core.moderator import DiscussionModerator
import asyncio

async def run_discussion():
    # Create agents
    agents = [
        ExpertAgent("PhysicsExpert", "quantum physics"),
        ExpertAgent("BiologyExpert", "molecular biology"),
        ExpertAgent("AIResearcher", "machine learning"),
        ExpertAgent("ChemistryExpert", "materials science")
    ]

    consensus_agent = ConsensusAgent()

    # Run discussion
    moderator = DiscussionModerator(max_rounds=10)
    queue = asyncio.Queue()
    session_log = []

    results = await moderator.moderate_discussion(
        agents=agents,
        consensus_agent=consensus_agent,
        task="Design a bio-inspired quantum computer",
        queue=queue,
        session_log=session_log
    )

    return results

# Execute
results = asyncio.run(run_discussion())
print(f"Consensus reached: {results['consensus_reached']}")
print(f"Total rounds: {results['total_rounds']}")
```

### Staged Testing Approach

| Stage | Description | Expected Output |
|-------|-------------|-----------------|
| 0 | No books (empty DBs) | Generic discussion, votes functional, no citations |
| 1 | 1–2 PDFs per agent | Selective citations, more grounded reasoning |
| 2 | 5–10 PDFs per domain | Richer citation usage, higher domain detail |
| 3 | Full corpus | Highly grounded, credible interdisciplinary synthesis |

### Advanced Features

#### Custom Agent Configuration
```python
# Create custom domain expert
from src.agents.expert import ExpertAgent
from src.core.vector_db import FAISSVectorDB

# Create knowledge base
vector_db = FAISSVectorDB()
vector_db.load_from_pdf("data/economics.pdf")

# Create economics expert
economics_expert = ExpertAgent(
    name="EconomicsExpert",
    domain="macroeconomics, game theory, financial markets",
    vector_db=vector_db,
    temperature=0.75
)
```

#### WebSocket Streaming
```python
import websockets
import json

async def stream_discussion():
    uri = "ws://localhost:8000/ws/discussion"
    async with websockets.connect(uri) as websocket:
        # Start discussion
        await websocket.send(json.dumps({
            "action": "start_discussion",
            "task": "Design sustainable energy solution",
            "max_rounds": 5
        }))

        # Receive real-time updates
        async for message in websocket:
            data = json.loads(message)
            if data["type"] == "agent_message":
                print(f"{data['agent']}: {data['message']}")
            elif data["type"] == "discussion_complete":
                print(f"Consensus: {data['consensus_reached']}")
                break
```

#### Session Management
```python
from src.core.session import SessionManager

# Create session manager
session_manager = SessionManager()

# List all sessions
sessions = session_manager.list_sessions()
for session in sessions:
    print(f"Session {session['session_id']}: {session['task']}")

# Get specific session
session = session_manager.get_session("20240115_143022")
print(f"Messages: {len(session.session_log)}")
print(f"Consensus: {session.consensus_reached}")

# Export session
session.export_session("exports/session_20240115.json")
```

### Production Deployment

#### Docker Deployment
```bash
# Build Docker image
docker build -t ai-agent:latest .

# Run with GPU support
docker run --gpus all -p 8000:8000 -p 8501:8501 ai-agent:latest

# Using Docker Compose
docker-compose up -d
```

#### Kubernetes Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Scale deployment
kubectl scale deployment ai-agent-api --replicas=5

# Check status
kubectl get pods -n ai-agent
```

### Performance Optimization

#### Model Quantization
```python
# Enable 8-bit quantization for faster inference
from src.config import config

config.model_config.quantization = "8bit"
config.model_config.device = "cuda"
config.model_config.max_memory = "24GB"
```

#### Batch Processing
```python
# Process multiple discussions in parallel
async def batch_discussions(tasks):
    results = await asyncio.gather(*[
        run_discussion(task) for task in tasks
    ])
    return results

tasks = [
    "Design quantum computer",
    "Create fusion reactor",
    "Develop AGI system"
]

all_results = asyncio.run(batch_discussions(tasks))
```

### Monitoring & Analytics

```python
# Enable metrics collection
from src.monitoring import MetricsCollector

metrics = MetricsCollector()

# Track discussion metrics
metrics.record_discussion(
    session_id="abc123",
    duration=120.5,
    rounds=8,
    consensus_reached=True,
    avg_novelty=7.8,
    avg_feasibility=6.5
)

# Get analytics
stats = metrics.get_statistics()
print(f"Average consensus rate: {stats['consensus_rate']:.2%}")
print(f"Average rounds to consensus: {stats['avg_rounds']:.1f}")
```

## Original Sample Code (Legacy)
# AI Expert Multi-Agent Discussion Pipeline + Streamlit Dashboard

import asyncio
import re
import numpy as np
import faiss
from openai import AsyncOpenAI
import PyPDF2
import tiktoken
import streamlit as st
from threading import Thread
import time
import pandas as pd

# Initialize OpenAI
client = AsyncOpenAI()

# ---------------------------
# Helper functions & DB
# ---------------------------

def chunk_pdf(file_path, chunk_size=500):
    reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page_num, page in enumerate(reader.pages, start=1):
        text += page.extract_text() + "\n"
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i+chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append({"text": chunk_text, "source": f"{file_path}, page {page_num}"})
    return chunks

def embed_text(text):
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding

class FAISSVectorDB:
    def __init__(self, embedding_dim=1536):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.entries = []

    def add(self, embedding, text, source):
        self.index.add(np.array([embedding], dtype='float32'))
        self.entries.append({"text": text, "source": source})

    def similarity_search(self, query_embedding, top_k=5):
        if self.index.ntotal == 0:
            return []
        D, I = self.index.search(np.array([query_embedding], dtype='float32'), top_k)
        return [self.entries[i] for i in I[0]]

# ---------------------------
# Expert Agent
# ---------------------------

class ExpertAgent:
    def __init__(self, name, domain, vector_db, model="gpt-4o-mini", temperature=0.7):
        self.name = name
        self.domain = domain
        self.vector_db = vector_db
        self.model = model
        self.temperature = temperature

    async def retrieve_material(self, query, top_k=5):
        if self.vector_db.index.ntotal == 0:
            return "", []
        embedding = embed_text(query)
        results = self.vector_db.similarity_search(embedding, top_k)
        formatted = "\n\n".join([f"{entry['text']} (Source: {entry['source']})" for entry in results])
        return formatted, results

    async def respond(self, queue, task, history, session_log):
        material_text, citations = await self.retrieve_material(history)
        system_prompt = f"""
You are {self.name}, a world-class expert in {self.domain}.
Use the following reference material if available:
{material_text if material_text else 'No relevant sources available at this point.'}

End your response with votes: Novelty X/10, Feasibility Y/10
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}\nConversation:\n{history}"}
        ]
        resp = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        content = resp.choices[0].message.content.strip()
        await queue.put((self.name, content, citations))
        session_log.append({"agent": self.name, "message": content, "citations": citations})

# ---------------------------
# Consensus Agent
# ---------------------------

class ConsensusAgent:
    def __init__(self, name, model="gpt-4o-mini", temperature=0.2):
        self.name = name
        self.model = model
        self.temperature = temperature

    async def respond(self, queue, task, history, session_log):
        novelty_scores = [int(n) for msg in [e['message'] for e in session_log] for n in re.findall(r"Novelty:\s*(\d+)/10", msg)]
        feasibility_scores = [int(f) for msg in [e['message'] for e in session_log] for f in re.findall(r"Feasibility:\s*(\d+)/10", msg)]

        novelty_avg = sum(novelty_scores)/len(novelty_scores) if novelty_scores else 0
        feasibility_avg = sum(feasibility_scores)/len(feasibility_scores) if feasibility_scores else 0

        novelty_threshold = 7
        feasibility_threshold = 6

        if novelty_avg >= novelty_threshold and feasibility_avg >= feasibility_threshold:
            decision_prompt = f"""consensus
- **Final Idea Summary** (include relevant citations)
- **Key Strengths** (cite supporting sources)
- **Feasibility Notes** (cite supporting sources)
- **Next Steps for Implementation**
"""
        else:
            decision_prompt = f"continue — discussion not ready. Avg Novelty: {novelty_avg:.1f}, Feasibility: {feasibility_avg:.1f}"

        messages = [
            {"role": "system", "content": f"You are {self.name}, cross-disciplinary synthesis expert."},
            {"role": "user", "content": f"Task: {task}\nConversation:\n{history}\nDecision:\n{decision_prompt}"}
        ]
        resp = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        content = resp.choices[0].message.content.strip()
        await queue.put((self.name, content, []))
        session_log.append({"agent": self.name, "message": content, "citations": []})

# ---------------------------
# Moderator
# ---------------------------

async def moderator(queue, task, agents, consensus_agent, session_log, max_rounds=10):
    conversation = []
    for _ in range(max_rounds):
        name, msg, citations = await queue.get()
        conversation.append((name, msg))
        session_log.append({"agent": name, "message": msg, "citations": citations})
        if name == "ConsensusAgent":
            if msg.lower().startswith("consensus"):
                break
            else:
                history_text = "\n".join([f"{n}: {m}" for n, m in conversation])
                await asyncio.gather(*[agent.respond(queue, task, history_text, session_log) for agent in agents])
        else:
            history_text = "\n".join([f"{n}: {m}" for n, m in conversation])
            await consensus_agent.respond(queue, task, history_text, session_log)

# ---------------------------
# Streamlit Dashboard
# ---------------------------

st.title("AI Expert Chatroom – Enhanced Dashboard")
st.sidebar.header("Session Settings")
task = st.sidebar.text_area("Task:", "Design a novel, scientifically grounded multi-modal search engine.")
max_rounds = st.sidebar.slider("Max Discussion Rounds:", 1, 20, 10)

conversation_box = st.empty()
votes_box = st.empty()
status_box = st.empty()
citations_box = st.empty()
timeline_box = st.empty()

# Session log
session_log = []

# Placeholder: empty DBs for testing
dummy_db = FAISSVectorDB()
agents = [
    ExpertAgent("PhysicsExpert", "Physics", dummy_db),
    ExpertAgent("BiologyExpert", "Biology", dummy_db),
    ExpertAgent("AIResearcher", "AI / ML", dummy_db),
    ExpertAgent("ChemistryExpert", "Chemistry", dummy_db),
]
consensus_agent = ConsensusAgent("ConsensusAgent")

queue = asyncio.Queue()

def run_async_session():
    asyncio.run(async_main())

async def async_main():
    await agents[0].respond(queue, task, "", session_log)
    await moderator(queue, task, agents, consensus_agent, session_log, max_rounds=max_rounds)

if st.button("Start Discussion"):
    thread = Thread(target=run_async_session, daemon=True)
    thread.start()

    while thread.is_alive():
        if session_log:
            conv_md = ""
            agent_colors = {"PhysicsExpert": "cyan", "BiologyExpert": "green",
                            "AIResearcher": "magenta", "ChemistryExpert": "orange",
                            "ConsensusAgent": "yellow"}
            for entry in session_log:
                color = agent_colors.get(entry["agent"], "white")
                conv_md += f"<span style='color:{color}'><b>{entry['agent']}</b>:</span> {entry['message']}<br><br>"
            conversation_box.markdown(conv_md, unsafe_allow_html=True)

            novelty_scores = [int(n) for msg in [e['message'] for e in session_log] for n in re.findall(r"Novelty:\s*(\d+)/10", msg)]
            feasibility_scores = [int(f) for msg in [e['message'] for e in session_log] for f in re.findall(r"Feasibility:\s*(\d+)/10", msg)]
            if novelty_scores and feasibility_scores:
                votes_box.text(f"Average Novelty: {sum(novelty_scores)/len(novelty_scores):.1f}, "
                               f"Average Feasibility: {sum(feasibility_scores)/len(feasibility_scores):.1f}")

            last_msg = session_log[-1]["message"]
            if "consensus" in last_msg.lower():
                status_box.success("✅ Consensus reached!")
            else:
                status_box.info("Discussion ongoing...")

            last_citations = session_log[-1].get("citations", [])
            if last_citations:
                citations_text = "\n".join([f"- {c['source']}" for c in last_citations])
            else:
                citations_text = "No sources cited."
            citations_box.text(citations_text)
        time.sleep(1)

Notes

You can run without books initially — agents will reason conceptually.

Gradually add PDFs to FAISS DBs for domain-specific grounding.

Session logs, votes, and citations allow tracking discussion quality and convergence.
- we definately not gonna  Option 3: Continue with Our Hack.