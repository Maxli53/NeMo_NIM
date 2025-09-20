# Multi-Agent Integration with GPT-OSS-20B

## Overview

This document details the integration of the Multi-Agent Discussion System with the native GPT-OSS-20B MoE implementation, replacing the mock LLM with actual model inference.

## Integration Architecture

```python
┌──────────────────────────────────────────────────────┐
│                  Agent Layer                          │
│   ExpertAgent → ConsensusAgent → Moderator          │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│              Integration Layer                        │
│         AgentModelAdapter (New Component)            │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│             Native MoE Backend                        │
│   NativeMoE → ExpertCache → Optimizations           │
└──────────────────────────────────────────────────────┘
```

## Implementation

### 1. Agent Model Adapter

```python
# agent_model_adapter.py
from native_moe_complete import NativeMoE
from moe_config import MoEConfig
import torch

class AgentModelAdapter:
    """
    Adapter to connect agents with native MoE model
    """
    def __init__(self, config: MoEConfig):
        self.config = config
        self.model = NativeMoE(config)
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        """Load GPT-OSS tokenizer"""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate response using native MoE
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt"
        ).to(self.config.device)

        # Generate using native MoE
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )

        # Decode response
        response = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        # Extract only new tokens
        response = response[len(prompt):]

        return response

    def get_expert_stats(self):
        """Get expert usage statistics"""
        return {
            'cache_hits': self.model.cache_manager.hits,
            'cache_misses': self.model.cache_manager.misses,
            'active_experts': len(self.model.cache_manager.cache),
            'memory_usage': torch.cuda.memory_allocated() / 1e9
        }
```

### 2. Updated Expert Agent

```python
# src/agents/expert.py (updated)
class ExpertAgent:
    def __init__(
        self,
        name: str,
        domain: str,
        vector_db: FAISSVectorDB,
        model_adapter: AgentModelAdapter,  # NEW
        temperature: float = 0.7
    ):
        self.name = name
        self.domain = domain
        self.vector_db = vector_db
        self.model = model_adapter  # Use adapter instead of API
        self.temperature = temperature

    async def respond(self, task: str, history: str) -> Dict:
        """Generate response using native MoE"""

        # Retrieve relevant knowledge
        material, citations = await self.retrieve_material(history)

        # Construct prompt
        prompt = self._construct_prompt(task, history, material)

        # Generate using native MoE
        response = await self.model.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=self.temperature
        )

        # Extract votes
        novelty, feasibility = self._extract_votes(response)

        return {
            'agent': self.name,
            'response': response,
            'citations': citations,
            'novelty': novelty,
            'feasibility': feasibility
        }

    def _construct_prompt(self, task, history, material):
        """Build agent-specific prompt"""
        return f"""You are {self.name}, a world-class expert in {self.domain}.

Task: {task}

Reference Material:
{material if material else 'No specific references available.'}

Conversation History:
{history}

Provide your expert perspective. End with:
Novelty: X/10
Feasibility: Y/10

Response:"""
```

### 3. Integration Configuration

```python
# multi_agent_config.yaml
multi_agent:
  # Model backend settings
  backend:
    type: "native_moe"
    model_path: "./gpt-oss-20b"
    device: "cuda"

  # Native MoE optimizations
  optimizations:
    cuda_kernels: true
    async_io: true
    cache_mode: "tiered"
    multi_gpu: false

  # Agent settings
  agents:
    physics:
      domain: "quantum mechanics, thermodynamics"
      temperature: 0.7
      knowledge_base: "data/physics_papers.pdf"

    biology:
      domain: "molecular biology, neuroscience"
      temperature: 0.75
      knowledge_base: "data/biology_papers.pdf"

    ai_ml:
      domain: "machine learning, neural architectures"
      temperature: 0.8
      knowledge_base: "data/ml_papers.pdf"

    chemistry:
      domain: "materials science, catalysis"
      temperature: 0.7
      knowledge_base: "data/chemistry_papers.pdf"

  # Consensus settings
  consensus:
    novelty_threshold: 7
    feasibility_threshold: 6
    max_rounds: 10

  # Performance settings
  performance:
    agent_timeout: 30
    parallel_agents: true
    batch_size: 4
    cache_responses: true
```

### 4. Complete Integration Script

```python
# multi_agent_integration.py
import asyncio
from pathlib import Path
from typing import List, Dict

from moe_config import MoEConfig
from agent_model_adapter import AgentModelAdapter
from src.agents.expert import ExpertAgent
from src.agents.consensus import ConsensusAgent
from src.core.moderator import DiscussionModerator
from src.core.vector_db import FAISSVectorDB

class MultiAgentWithGPTOSS:
    """
    Complete multi-agent system with GPT-OSS backend
    """

    def __init__(self, config_path: str = "multi_agent_config.yaml"):
        self.config = self._load_config(config_path)
        self.model_adapter = self._initialize_model()
        self.agents = self._initialize_agents()
        self.consensus_agent = self._initialize_consensus()
        self.moderator = DiscussionModerator(
            max_rounds=self.config['consensus']['max_rounds']
        )

    def _load_config(self, path: str) -> Dict:
        """Load configuration from YAML"""
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_model(self) -> AgentModelAdapter:
        """Initialize native MoE with optimizations"""
        moe_config = MoEConfig()

        # Set model path
        moe_config.model_path = self.config['backend']['model_path']
        moe_config.device = self.config['backend']['device']

        # Enable optimizations
        opts = self.config['optimizations']
        moe_config.cuda_kernels.enabled = opts.get('cuda_kernels', False)
        moe_config.async_io.enabled = opts.get('async_io', False)
        moe_config.cache.mode = opts.get('cache_mode', 'single')
        moe_config.multi_gpu.enabled = opts.get('multi_gpu', False)

        return AgentModelAdapter(moe_config)

    def _initialize_agents(self) -> List[ExpertAgent]:
        """Initialize expert agents with knowledge bases"""
        agents = []

        for agent_name, agent_config in self.config['agents'].items():
            # Create knowledge base
            vector_db = FAISSVectorDB()

            # Load PDFs if specified
            kb_path = agent_config.get('knowledge_base')
            if kb_path and Path(kb_path).exists():
                vector_db.load_from_pdf(kb_path)

            # Create agent
            agent = ExpertAgent(
                name=f"{agent_name.capitalize()}Expert",
                domain=agent_config['domain'],
                vector_db=vector_db,
                model_adapter=self.model_adapter,
                temperature=agent_config['temperature']
            )

            agents.append(agent)

        return agents

    def _initialize_consensus(self) -> ConsensusAgent:
        """Initialize consensus agent"""
        return ConsensusAgent(
            model_adapter=self.model_adapter,
            novelty_threshold=self.config['consensus']['novelty_threshold'],
            feasibility_threshold=self.config['consensus']['feasibility_threshold']
        )

    async def discuss(self, task: str) -> Dict:
        """
        Run multi-agent discussion on task
        """
        print(f"Starting discussion: {task}")

        # Initialize session
        session_log = []
        start_time = asyncio.get_event_loop().time()

        # Run moderated discussion
        result = await self.moderator.moderate_discussion(
            agents=self.agents,
            consensus_agent=self.consensus_agent,
            task=task,
            session_log=session_log
        )

        # Calculate metrics
        duration = asyncio.get_event_loop().time() - start_time
        model_stats = self.model_adapter.get_expert_stats()

        # Compile results
        return {
            'task': task,
            'consensus_reached': result['consensus'],
            'rounds': result['rounds'],
            'final_novelty': result['novelty'],
            'final_feasibility': result['feasibility'],
            'synthesis': result['synthesis'],
            'session_log': session_log,
            'duration': duration,
            'model_stats': model_stats
        }

    def benchmark(self, tasks: List[str]) -> Dict:
        """
        Benchmark system on multiple tasks
        """
        results = []

        for task in tasks:
            result = asyncio.run(self.discuss(task))
            results.append(result)

        # Calculate aggregate metrics
        return {
            'total_tasks': len(tasks),
            'consensus_rate': sum(1 for r in results if r['consensus_reached']) / len(results),
            'avg_rounds': sum(r['rounds'] for r in results) / len(results),
            'avg_duration': sum(r['duration'] for r in results) / len(results),
            'cache_efficiency': self.model_adapter.get_expert_stats()
        }
```

## Performance Optimizations

### 1. Agent Response Caching

```python
class ResponseCache:
    """Cache agent responses for similar queries"""

    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size

    def get_key(self, agent, task, context):
        """Generate cache key"""
        return hash((agent, task, context[:100]))

    def get(self, agent, task, context):
        """Retrieve cached response"""
        key = self.get_key(agent, task, context)
        return self.cache.get(key)

    def store(self, agent, task, context, response):
        """Store response in cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest = min(self.cache.keys())
            del self.cache[oldest]

        key = self.get_key(agent, task, context)
        self.cache[key] = response
```

### 2. Parallel Agent Processing

```python
async def process_agents_parallel(agents, task, history, model_adapter):
    """
    Process all agents in parallel using shared model
    """
    # Create tasks for each agent
    tasks = []
    for agent in agents:
        task = asyncio.create_task(
            agent.respond(task, history)
        )
        tasks.append(task)

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    return responses
```

### 3. Expert Preloading

```python
def preload_common_experts(model: NativeMoE, common_expert_ids: List[int]):
    """
    Preload frequently used experts
    """
    for layer_idx in range(model.config.num_layers):
        for expert_idx in common_expert_ids:
            # Load into cache
            model.cache_manager.get_expert(layer_idx, expert_idx)

    print(f"Preloaded {len(common_expert_ids)} experts per layer")
```

## Testing

### Integration Test

```python
# test_integration.py
import pytest
import asyncio

@pytest.mark.asyncio
async def test_multi_agent_with_native_moe():
    """Test complete integration"""

    # Initialize system
    system = MultiAgentWithGPTOSS("test_config.yaml")

    # Run discussion
    result = await system.discuss(
        "Design a quantum computer using biological principles"
    )

    # Verify results
    assert 'consensus_reached' in result
    assert 'final_novelty' in result
    assert 'final_feasibility' in result
    assert len(result['session_log']) > 0

    # Check model stats
    stats = result['model_stats']
    assert stats['cache_hits'] > 0
    assert stats['memory_usage'] < 5.0  # GB

@pytest.mark.benchmark
def test_performance():
    """Benchmark integrated system"""

    system = MultiAgentWithGPTOSS()

    tasks = [
        "Design fusion reactor",
        "Create AGI system",
        "Develop nanorobots"
    ]

    results = system.benchmark(tasks)

    assert results['consensus_rate'] > 0.5
    assert results['avg_duration'] < 60  # seconds
```

## Deployment

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  multi-agent:
    build: .
    image: multi-agent-gpt-oss:latest
    ports:
      - "8000:8000"  # API
      - "8501:8501"  # Streamlit
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_PATH=/app/models/gpt-oss-20b
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Production Configuration

```yaml
# production_config.yaml
multi_agent:
  backend:
    type: "native_moe"
    model_path: "/models/gpt-oss-20b"
    device: "cuda"

  optimizations:
    cuda_kernels: true      # Validated in staging
    async_io: true          # Reduces latency
    cache_mode: "tiered"    # Better hit rate
    multi_gpu: false        # Enable if >1 GPU

  performance:
    max_concurrent_discussions: 5
    agent_timeout: 45
    response_cache_size: 500
    preload_common_experts: true

  monitoring:
    enable_metrics: true
    prometheus_port: 9090
    log_level: "INFO"
```

## Monitoring

### Key Metrics

```python
INTEGRATION_METRICS = {
    'discussion_latency': Histogram(
        'ma_discussion_latency_seconds',
        'Time to complete discussion'
    ),
    'agent_response_time': Histogram(
        'ma_agent_response_time_seconds',
        'Individual agent response time',
        ['agent_name']
    ),
    'model_cache_hit_rate': Gauge(
        'ma_model_cache_hit_rate',
        'Native MoE cache hit rate'
    ),
    'consensus_success_rate': Gauge(
        'ma_consensus_rate',
        'Rate of successful consensus'
    ),
    'gpu_memory_usage': Gauge(
        'ma_gpu_memory_gb',
        'GPU memory usage in GB'
    ),
}
```

## Troubleshooting

### Common Issues

1. **OOM Errors**
```python
# Reduce batch size
config.performance.batch_size = 2

# Reduce cache size
config.optimizations.cache_size_gb = 1.0

# Enable gradient checkpointing
config.optimizations.gradient_checkpointing = True
```

2. **Slow Agent Responses**
```python
# Enable optimizations
config.optimizations.cuda_kernels = True
config.optimizations.async_io = True

# Reduce max tokens
config.agents.max_tokens = 300
```

3. **Poor Consensus Rate**
```python
# Adjust thresholds
config.consensus.novelty_threshold = 6
config.consensus.feasibility_threshold = 5

# Increase max rounds
config.consensus.max_rounds = 15
```

## Conclusion

The integration of the Multi-Agent Discussion System with the native GPT-OSS-20B MoE implementation provides a powerful, efficient platform for interdisciplinary problem-solving. The combination of optimized inference and intelligent agent orchestration enables complex discussions at scale.

---

*For multi-agent overview, see [40_MULTI_AGENT_OVERVIEW.md](40_MULTI_AGENT_OVERVIEW.md)*
*For native MoE details, see [11_NATIVE_MOE_IMPLEMENTATION.md](11_NATIVE_MOE_IMPLEMENTATION.md)*