# 📡 API Documentation

## Table of Contents
- [Overview](#overview)
- [Authentication](#authentication)
- [REST API Endpoints](#rest-api-endpoints)
- [WebSocket API](#websocket-api)
- [MCP Tools](#mcp-tools)
- [Python SDK](#python-sdk)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

## Overview

The Multi-Agent Discussion System provides multiple API interfaces for integration:

1. **REST API**: Standard HTTP endpoints for discussion management
2. **WebSocket API**: Real-time bidirectional communication
3. **MCP Protocol**: Claude Desktop integration
4. **Python SDK**: Native Python integration

### Base URLs

```
REST API:    http://localhost:8000/api/v1
WebSocket:   ws://localhost:8000/ws
MCP Server:  http://localhost:8000/tools
Health:      http://localhost:8000/health
```

## Authentication

### API Key Authentication

```http
Authorization: Bearer YOUR_API_KEY
```

### Python Example
```python
import requests

headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/api/v1/discussion",
    headers=headers,
    json={"task": "Design a quantum computer"}
)
```

## REST API Endpoints

### 1. Start Discussion

**Endpoint**: `POST /api/v1/discussion`

**Description**: Initiates a new multi-agent discussion

**Request Body**:
```json
{
  "task": "string",              // Required: The discussion task
  "max_rounds": 5,                // Optional: Maximum discussion rounds (default: 5)
  "use_knowledge_base": true,     // Optional: Enable RAG (default: true)
  "pdf_paths": [                  // Optional: Custom PDFs for each agent
    "physics.pdf",
    "biology.pdf",
    "ai_ml.pdf",
    "chemistry.pdf"
  ],
  "agent_config": {               // Optional: Override agent settings
    "temperature": 0.7,
    "top_k_retrieval": 5
  }
}
```

**Response**:
```json
{
  "session_id": "20240101_120000",
  "task": "Design a quantum computer",
  "status": "in_progress",
  "consensus_reached": false,
  "total_rounds": 3,
  "total_messages": 12,
  "avg_novelty": 7.5,
  "avg_feasibility": 6.8,
  "messages": [
    {
      "agent": "PhysicsExpert",
      "message": "From a physics perspective...",
      "citations": [
        {
          "source": "quantum_mechanics.pdf, page 42",
          "text": "Quantum superposition allows..."
        }
      ],
      "novelty_score": 8,
      "feasibility_score": 7,
      "timestamp": "2024-01-01T12:00:00Z"
    }
  ],
  "final_consensus": null
}
```

**Status Codes**:
- `200 OK`: Discussion completed successfully
- `202 Accepted`: Discussion started (async mode)
- `400 Bad Request`: Invalid parameters
- `500 Internal Server Error`: Server error

### 2. Get Session

**Endpoint**: `GET /api/v1/session/{session_id}`

**Description**: Retrieves details of a specific discussion session

**Path Parameters**:
- `session_id`: The session identifier (e.g., "20240101_120000")

**Response**:
```json
{
  "session_id": "20240101_120000",
  "task": "Design a quantum computer",
  "start_time": "2024-01-01T12:00:00Z",
  "end_time": "2024-01-01T12:15:00Z",
  "status": "completed",
  "statistics": {
    "total_messages": 15,
    "expert_messages": 12,
    "consensus_messages": 3,
    "total_citations": 45,
    "avg_novelty": 7.5,
    "avg_feasibility": 6.8,
    "duration_seconds": 900
  },
  "messages": [...],
  "final_consensus": "Based on our interdisciplinary discussion..."
}
```

### 3. List Sessions

**Endpoint**: `GET /api/v1/sessions`

**Description**: Lists all discussion sessions

**Query Parameters**:
- `limit`: Maximum number of sessions (default: 20)
- `offset`: Pagination offset (default: 0)
- `status`: Filter by status (in_progress, completed, failed)
- `start_date`: Filter sessions after date (ISO 8601)
- `end_date`: Filter sessions before date (ISO 8601)

**Response**:
```json
{
  "sessions": [
    {
      "session_id": "20240101_120000",
      "task": "Design a quantum computer",
      "start_time": "2024-01-01T12:00:00Z",
      "status": "completed",
      "message_count": 15,
      "consensus_reached": true
    }
  ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

### 4. Stop Discussion

**Endpoint**: `POST /api/v1/session/{session_id}/stop`

**Description**: Stops an ongoing discussion

**Response**:
```json
{
  "session_id": "20240101_120000",
  "status": "stopped",
  "messages_generated": 8,
  "reason": "User requested stop"
}
```

### 5. Export Session

**Endpoint**: `GET /api/v1/session/{session_id}/export`

**Description**: Exports session data in various formats

**Query Parameters**:
- `format`: Export format (json, csv, pdf) (default: json)

**Response**:
- JSON: Application/json
- CSV: Text/csv
- PDF: Application/pdf

### 6. Update Agent Knowledge

**Endpoint**: `POST /api/v1/agent/{agent_name}/knowledge`

**Description**: Updates an agent's knowledge base with new documents

**Request Body**:
```json
{
  "pdf_url": "https://example.com/document.pdf",
  "pdf_base64": "...",  // Alternative: base64 encoded PDF
  "process_options": {
    "chunk_size": 500,
    "chunk_overlap": 50
  }
}
```

**Response**:
```json
{
  "agent": "PhysicsExpert",
  "chunks_added": 150,
  "embedding_time_ms": 2500,
  "index_updated": true
}
```

### 7. Agent Statistics

**Endpoint**: `GET /api/v1/agent/{agent_name}/stats`

**Description**: Gets statistics for a specific agent

**Response**:
```json
{
  "agent": "PhysicsExpert",
  "domain": "Physics and Quantum Mechanics",
  "knowledge_base": {
    "total_chunks": 1500,
    "unique_sources": 5,
    "index_size_mb": 12.5,
    "last_updated": "2024-01-01T10:00:00Z"
  },
  "performance": {
    "avg_response_time_ms": 850,
    "total_responses": 250,
    "avg_novelty_score": 7.8,
    "avg_feasibility_score": 7.2
  }
}
```

### 8. Health Check

**Endpoint**: `GET /health`

**Description**: System health status

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "components": {
    "model": {
      "status": "healthy",
      "model_name": "gpt-oss-20b",
      "device": "cuda",
      "memory_usage_gb": 18.5
    },
    "embedding": {
      "status": "healthy",
      "model": "all-MiniLM-L6-v2",
      "cache_size": 5000
    },
    "vector_db": {
      "status": "healthy",
      "total_indices": 4,
      "total_entries": 6000
    },
    "sessions": {
      "status": "healthy",
      "active_sessions": 2,
      "total_sessions": 150
    }
  }
}
```

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/discussion');

ws.onopen = () => {
  console.log('Connected to discussion server');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from server');
};
```

### Message Types

#### 1. Start Discussion

**Client → Server**:
```json
{
  "action": "start_discussion",
  "task": "Design a sustainable energy system",
  "max_rounds": 5,
  "use_knowledge_base": true
}
```

**Server → Client**:
```json
{
  "type": "session_started",
  "session_id": "20240101_120000",
  "task": "Design a sustainable energy system",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### 2. Agent Message

**Server → Client**:
```json
{
  "type": "agent_message",
  "session_id": "20240101_120000",
  "round": 1,
  "agent": "PhysicsExpert",
  "message": "From a physics perspective...",
  "citations": [...],
  "novelty_score": 8,
  "feasibility_score": 7,
  "timestamp": "2024-01-01T12:00:15Z"
}
```

#### 3. Consensus Update

**Server → Client**:
```json
{
  "type": "consensus_update",
  "session_id": "20240101_120000",
  "consensus_reached": false,
  "avg_novelty": 7.5,
  "avg_feasibility": 6.8,
  "rounds_remaining": 3
}
```

#### 4. Discussion Complete

**Server → Client**:
```json
{
  "type": "discussion_complete",
  "session_id": "20240101_120000",
  "consensus_reached": true,
  "total_rounds": 5,
  "total_messages": 20,
  "final_consensus": "Based on our discussion...",
  "duration_seconds": 300
}
```

#### 5. Error

**Server → Client**:
```json
{
  "type": "error",
  "error_code": "AGENT_TIMEOUT",
  "message": "Agent failed to respond within timeout",
  "session_id": "20240101_120000",
  "recoverable": true
}
```

### WebSocket Commands

#### Stop Discussion
```json
{
  "action": "stop_discussion",
  "session_id": "20240101_120000"
}
```

#### Subscribe to Updates
```json
{
  "action": "subscribe",
  "session_id": "20240101_120000",
  "events": ["agent_message", "consensus_update"]
}
```

#### Unsubscribe
```json
{
  "action": "unsubscribe",
  "session_id": "20240101_120000"
}
```

## MCP Tools

### Tool Registration

The MCP server exposes the following tools for Claude Desktop:

#### 1. run_discussion

**Description**: Run a multi-agent expert discussion

**Parameters**:
```typescript
{
  task: string;              // Required: Discussion task
  max_rounds?: number;        // Optional: Max rounds (default: 5)
  pdf_paths?: string[];       // Optional: PDF paths for agents
  use_knowledge_base?: boolean; // Optional: Enable RAG (default: true)
}
```

**Example in Claude Desktop**:
```
@multiAgentDiscussion.run_discussion task="Design a quantum neural network" max_rounds=7
```

#### 2. get_session

**Description**: Get details of a discussion session

**Parameters**:
```typescript
{
  session_id: string;  // Required: Session identifier
}
```

**Example**:
```
@multiAgentDiscussion.get_session session_id="20240101_120000"
```

#### 3. list_sessions

**Description**: List all discussion sessions

**Parameters**:
```typescript
{
  limit?: number;      // Optional: Max results (default: 10)
  status?: string;     // Optional: Filter by status
}
```

**Example**:
```
@multiAgentDiscussion.list_sessions limit=5 status="completed"
```

#### 4. ping

**Description**: Health check for MCP server

**Parameters**: None

**Example**:
```
@multiAgentDiscussion.ping
```

### MCP Configuration

Configure Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "multiAgentDiscussion": {
      "command": "python",
      "args": [
        "C:/path/to/AI_agents/main.py",
        "--mode",
        "mcp"
      ],
      "env": {
        "PYTHONPATH": "C:/path/to/AI_agents",
        "MODEL_DEVICE": "cuda"
      }
    }
  }
}
```

## Python SDK

### Installation

```python
pip install ai-agents-sdk  # If published
# Or local installation
pip install -e /path/to/AI_agents
```

### Basic Usage

```python
from ai_agents import MultiAgentSystem, DiscussionConfig

# Initialize system
system = MultiAgentSystem()
await system.initialize()

# Configure discussion
config = DiscussionConfig(
    task="Design a sustainable city",
    max_rounds=10,
    use_knowledge_base=True,
    novelty_threshold=7.0,
    feasibility_threshold=6.0
)

# Run discussion
results = await system.run_discussion(config)

# Access results
print(f"Consensus: {results.consensus_reached}")
print(f"Novelty: {results.avg_novelty}")
print(f"Feasibility: {results.avg_feasibility}")

# Export session
session_path = results.export_session("output.json")
```

### Advanced Usage

#### Custom Agent Configuration

```python
from ai_agents import ExpertAgent, AgentConfig

# Create custom agent
quantum_expert = ExpertAgent(
    name="QuantumExpert",
    domain="Quantum Computing",
    config=AgentConfig(
        temperature=0.6,
        top_k_retrieval=10,
        pdf_path="quantum_physics.pdf"
    )
)

# Add to system
system.add_agent(quantum_expert)
```

#### Event Handlers

```python
from ai_agents import EventHandler

class CustomHandler(EventHandler):
    async def on_agent_message(self, agent: str, message: str, citations: List):
        print(f"{agent}: {message[:100]}...")

    async def on_consensus_update(self, consensus: bool, scores: Dict):
        print(f"Consensus: {consensus}, Scores: {scores}")

    async def on_error(self, error: Exception):
        print(f"Error: {error}")

# Register handler
system.add_event_handler(CustomHandler())
```

#### Batch Processing

```python
# Process multiple discussions
tasks = [
    "Design a quantum computer",
    "Create sustainable energy",
    "Develop AGI safety measures"
]

results = await system.batch_discussions(tasks, max_parallel=3)

for task, result in zip(tasks, results):
    print(f"{task}: Consensus={result.consensus_reached}")
```

#### Custom Consensus Strategy

```python
from ai_agents import ConsensusStrategy

class WeightedConsensus(ConsensusStrategy):
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

    def evaluate(self, votes: List[Dict]) -> Tuple[bool, float, float]:
        # Custom weighted voting logic
        weighted_novelty = sum(
            v['novelty'] * self.weights.get(v['agent'], 1.0)
            for v in votes
        )
        # ... calculate consensus
        return consensus, avg_novelty, avg_feasibility

# Use custom strategy
system.set_consensus_strategy(WeightedConsensus({
    "PhysicsExpert": 1.5,
    "BiologyExpert": 1.2,
    "AIResearcher": 1.3,
    "ChemistryExpert": 1.1
}))
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "AGENT_TIMEOUT",
    "message": "Agent PhysicsExpert failed to respond within 30 seconds",
    "details": {
      "agent": "PhysicsExpert",
      "timeout_seconds": 30,
      "round": 3
    },
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_123456"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Invalid request parameters |
| `TASK_TOO_LONG` | 400 | Task exceeds maximum length |
| `SESSION_NOT_FOUND` | 404 | Session ID not found |
| `AGENT_NOT_FOUND` | 404 | Agent name not found |
| `MODEL_ERROR` | 500 | Model inference failed |
| `EMBEDDING_ERROR` | 500 | Embedding generation failed |
| `VECTOR_DB_ERROR` | 500 | Vector database error |
| `AGENT_TIMEOUT` | 504 | Agent response timeout |
| `CONSENSUS_TIMEOUT` | 504 | Consensus evaluation timeout |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `QUOTA_EXCEEDED` | 402 | Usage quota exceeded |
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `FORBIDDEN` | 403 | Insufficient permissions |

### Retry Strategy

```python
import backoff
import requests

@backoff.on_exception(
    backoff.expo,
    requests.exceptions.RequestException,
    max_tries=3,
    max_time=60
)
def call_api_with_retry(url, data):
    response = requests.post(url, json=data, timeout=30)
    response.raise_for_status()
    return response.json()
```

## Rate Limiting

### Default Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/api/v1/discussion` | 10 requests | 1 minute |
| `/api/v1/session/*` | 100 requests | 1 minute |
| `/api/v1/sessions` | 50 requests | 1 minute |
| `/ws/discussion` | 5 connections | 1 minute |
| `/health` | 1000 requests | 1 minute |

### Rate Limit Headers

```http
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1704110460
Retry-After: 45
```

### Handling Rate Limits

```python
def handle_rate_limit(response):
    if response.status_code == 429:
        retry_after = int(response.headers.get('Retry-After', 60))
        print(f"Rate limited. Retrying after {retry_after} seconds")
        time.sleep(retry_after)
        return True
    return False
```

## Examples

### Complete Discussion Flow

```python
import asyncio
import aiohttp

async def run_complete_discussion():
    async with aiohttp.ClientSession() as session:
        # 1. Start discussion
        async with session.post(
            'http://localhost:8000/api/v1/discussion',
            json={
                "task": "Design a brain-computer interface",
                "max_rounds": 7,
                "use_knowledge_base": True
            }
        ) as resp:
            result = await resp.json()
            session_id = result['session_id']

        # 2. Monitor progress (WebSocket)
        async with session.ws_connect('ws://localhost:8000/ws/discussion') as ws:
            await ws.send_json({
                "action": "subscribe",
                "session_id": session_id
            })

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data['type'] == 'discussion_complete':
                        break
                    print(f"{data['agent']}: {data['message'][:50]}...")

        # 3. Get final results
        async with session.get(
            f'http://localhost:8000/api/v1/session/{session_id}'
        ) as resp:
            final_result = await resp.json()

        # 4. Export session
        async with session.get(
            f'http://localhost:8000/api/v1/session/{session_id}/export?format=pdf'
        ) as resp:
            pdf_content = await resp.read()
            with open(f'{session_id}.pdf', 'wb') as f:
                f.write(pdf_content)

        return final_result

# Run
result = asyncio.run(run_complete_discussion())
print(f"Consensus reached: {result['statistics']['consensus_reached']}")
```

### Streaming Discussion with Server-Sent Events

```python
import sseclient
import requests

def stream_discussion(task):
    response = requests.post(
        'http://localhost:8000/api/v1/discussion/stream',
        json={"task": task, "max_rounds": 5},
        stream=True
    )

    client = sseclient.SSEClient(response)

    for event in client.events():
        data = json.loads(event.data)

        if event.event == 'agent_message':
            print(f"{data['agent']}: {data['message']}")
        elif event.event == 'consensus_update':
            print(f"Consensus: {data['consensus_reached']}")
        elif event.event == 'complete':
            print("Discussion complete!")
            break
```

### Batch API Calls

```python
async def batch_api_calls(tasks):
    async with aiohttp.ClientSession() as session:
        # Create all discussions concurrently
        create_tasks = []
        for task in tasks:
            create_tasks.append(
                session.post(
                    'http://localhost:8000/api/v1/discussion',
                    json={"task": task, "max_rounds": 3}
                )
            )

        responses = await asyncio.gather(*create_tasks)

        # Extract session IDs
        sessions = []
        for resp in responses:
            data = await resp.json()
            sessions.append(data['session_id'])

        # Wait for all to complete
        await asyncio.sleep(30)

        # Get all results
        result_tasks = []
        for session_id in sessions:
            result_tasks.append(
                session.get(f'http://localhost:8000/api/v1/session/{session_id}')
            )

        results = await asyncio.gather(*result_tasks)

        return [await r.json() for r in results]
```

---

This API documentation provides comprehensive coverage of all available endpoints, protocols, and integration methods for the Multi-Agent Discussion System.