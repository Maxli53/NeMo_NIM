# Native MoE Implementation Details

## Core Architecture

### Overview
The native MoE implementation replaces HuggingFace's approach of loading all 32 experts with a dynamic system that loads only the required top-4 experts per token, achieving 87.5% memory reduction.

## Key Components

### 1. Dynamic Expert Router
```python
class ExpertRouter(nn.Module):
    """
    Attention-based router that selects top-k experts per token
    """
    def __init__(self, hidden_dim=2880, num_experts=32, top_k=4):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, hidden_states):
        # Compute router logits
        router_logits = self.gate(hidden_states)  # [batch, seq, experts]

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )

        # Normalize with softmax
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_weights, top_k_indices
```

**Key Features:**
- Linear projection to expert scores
- Top-k selection (k=4 by default)
- Softmax normalization for weights
- Gradient flow through router for learning

### 2. Expert Cache Manager
```python
class ExpertCacheManager:
    """
    LRU cache for expert weights with dynamic loading
    """
    def __init__(self, cache_size_gb=2.0):
        self.cache = OrderedDict()
        self.max_size = int(cache_size_gb * 1e9)
        self.current_size = 0
        self.hits = 0
        self.misses = 0

    def get_expert(self, layer_idx, expert_idx):
        key = (layer_idx, expert_idx)

        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

        # Cache miss - load from disk
        self.misses += 1
        expert = self._load_expert_from_disk(layer_idx, expert_idx)

        # Add to cache with eviction if needed
        self._add_to_cache(key, expert)

        return expert
```

**Cache Strategy:**
- LRU (Least Recently Used) eviction
- Dynamic size management
- Hit rate tracking
- Thread-safe operations

### 3. Expert Mixer
```python
class ExpertMixer:
    """
    Combines outputs from selected experts using routing weights
    """
    def mix_expert_outputs(
        self,
        expert_outputs: Dict[int, torch.Tensor],
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor
    ):
        batch_size, seq_len, hidden_dim = expert_outputs[0].shape
        num_experts = expert_indices.shape[-1]

        # Initialize output tensor
        mixed_output = torch.zeros(
            batch_size, seq_len, hidden_dim,
            device=expert_weights.device,
            dtype=expert_outputs[0].dtype
        )

        # Weighted sum of expert outputs
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(num_experts):
                    expert_idx = expert_indices[b, s, k].item()
                    weight = expert_weights[b, s, k]

                    if expert_idx in expert_outputs:
                        expert_out = expert_outputs[expert_idx][b, s]
                        mixed_output[b, s] += weight * expert_out

        return mixed_output
```

**Mixing Process:**
1. Initialize zero tensor for output
2. Iterate through selected experts
3. Apply routing weights
4. Accumulate weighted outputs

### 4. SwiGLU Activation
```python
class SwiGLU(nn.Module):
    """
    SwiGLU activation function used in GPT-OSS experts
    """
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)
```

**Properties:**
- Gated linear unit variant
- Better gradient flow than ReLU
- Used in all expert FFN blocks

## Memory Management

### Memory Layout
```
GPU Memory (24GB Total)
├── Model Weights (Fixed)
│   ├── Embeddings: 0.5 GB
│   ├── Routers (24 layers): 1.94 MB
│   └── Output layers: 0.2 GB
│
├── Expert Cache (Dynamic)
│   ├── Active Experts (4): 105.6 MB
│   ├── Cached Experts: up to 2.0 GB
│   └── Eviction Buffer: 100 MB
│
├── Activations (Runtime)
│   ├── Hidden States: varies by batch
│   ├── Router Logits: minimal
│   └── Intermediate: varies
│
└── Free Memory: ~20 GB
```

### Memory Optimization Strategies

#### 1. Just-In-Time Loading
```python
def load_expert_on_demand(layer_idx, expert_idx):
    """Load expert only when needed"""
    expert_path = f"experts/layer_{layer_idx}/expert_{expert_idx}.safetensors"

    with safe_open(expert_path, framework="pt", device="cpu") as f:
        expert = {
            "up_proj": f.get_tensor("up_proj.weight"),
            "down_proj": f.get_tensor("down_proj.weight"),
            "gate_proj": f.get_tensor("gate_proj.weight")
        }

    # Move to GPU only when needed
    return {k: v.to("cuda") for k, v in expert.items()}
```

#### 2. Memory Pool Reuse
```python
class MemoryPool:
    """Reuse allocated tensors to reduce fragmentation"""
    def __init__(self):
        self.pools = {}

    def get_tensor(self, shape, dtype):
        key = (shape, dtype)
        if key not in self.pools:
            self.pools[key] = []

        if self.pools[key]:
            return self.pools[key].pop()

        return torch.empty(shape, dtype=dtype)

    def return_tensor(self, tensor):
        key = (tensor.shape, tensor.dtype)
        tensor.zero_()  # Clear data
        self.pools[key].append(tensor)
```

## Performance Optimizations

### 1. Batched Operations
```python
def batched_expert_forward(expert_batch, input_batch):
    """Process multiple tokens through same expert efficiently"""
    # Stack inputs for batch processing
    stacked_input = torch.cat(input_batch, dim=0)

    # Single forward pass
    output = expert_batch(stacked_input)

    # Split outputs back
    return torch.split(output, [x.size(0) for x in input_batch])
```

### 2. Parallel Expert Processing
```python
async def process_experts_parallel(expert_indices, hidden_states):
    """Process different experts in parallel"""
    tasks = []
    for expert_idx in expert_indices:
        task = asyncio.create_task(
            process_expert_async(expert_idx, hidden_states)
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return dict(zip(expert_indices, results))
```

### 3. Router Caching
```python
class RouterCache:
    """Cache router decisions for repeated inputs"""
    def __init__(self, cache_size=1000):
        self.cache = {}
        self.max_size = cache_size

    def get_routing(self, input_hash):
        return self.cache.get(input_hash)

    def store_routing(self, input_hash, routing_weights, routing_indices):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))

        self.cache[input_hash] = (routing_weights, routing_indices)
```

## Error Handling

### 1. Expert Loading Failures
```python
def safe_load_expert(layer_idx, expert_idx, max_retries=3):
    """Robust expert loading with retries"""
    for attempt in range(max_retries):
        try:
            return load_expert(layer_idx, expert_idx)
        except Exception as e:
            logger.warning(f"Failed to load expert {layer_idx}.{expert_idx}: {e}")
            if attempt == max_retries - 1:
                # Return zero expert as fallback
                return create_zero_expert()
            time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
```

### 2. OOM Protection
```python
def check_memory_before_load(expert_size):
    """Prevent OOM by checking available memory"""
    free_memory = torch.cuda.mem_get_info()[0]
    safety_margin = 0.1 * free_memory  # Keep 10% free

    if expert_size > (free_memory - safety_margin):
        # Trigger cache eviction
        evict_experts_to_free_space(expert_size)

    # Final check
    if expert_size > torch.cuda.mem_get_info()[0]:
        raise MemoryError("Insufficient GPU memory for expert")
```

## Validation & Testing

### 1. Numerical Validation
```python
def validate_moe_output(native_output, reference_output, tolerance=1e-5):
    """Compare native MoE with reference implementation"""
    diff = torch.abs(native_output - reference_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    assert max_diff < tolerance, f"Max diff {max_diff} exceeds tolerance"
    assert mean_diff < tolerance/10, f"Mean diff {mean_diff} too high"

    return True
```

### 2. Memory Leak Detection
```python
def check_memory_leaks(func, iterations=100):
    """Detect memory leaks in implementation"""
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()

    for _ in range(iterations):
        func()
        torch.cuda.empty_cache()

    final_memory = torch.cuda.memory_allocated()
    leak_per_iter = (final_memory - initial_memory) / iterations

    assert leak_per_iter < 1024, f"Memory leak detected: {leak_per_iter} bytes/iter"
```

## Integration Points

### 1. HuggingFace Compatibility
```python
class NativeMoEWrapper(nn.Module):
    """Wrapper for HuggingFace compatibility"""
    def __init__(self, native_moe):
        super().__init__()
        self.native_moe = native_moe

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Convert HF inputs to native format
        hidden_states = self.native_moe.embed(input_ids)

        # Process through native MoE
        output = self.native_moe(hidden_states)

        # Convert back to HF format
        return transformers.modeling_outputs.CausalLMOutputWithPast(
            logits=output,
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )
```

### 2. ONNX Export
```python
def export_to_onnx(model, output_path):
    """Export native MoE to ONNX format"""
    dummy_input = torch.randn(1, 128, 2880)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 1: 'sequence'},
            'output': {0: 'batch', 1: 'sequence'}
        }
    )
```

## Configuration Options

### Essential Parameters
```yaml
moe:
  num_experts: 32          # Total experts per layer
  experts_per_token: 4     # Top-k selection
  expert_capacity: 1.0     # Load balancing factor

cache:
  strategy: "lru"          # Eviction strategy
  size_gb: 2.0            # GPU cache size
  prefetch: true          # Enable prefetching

performance:
  batch_operations: true   # Batch similar operations
  parallel_experts: true   # Process experts in parallel
  router_cache: false     # Cache routing decisions
```

## Debugging Tools

### 1. Expert Access Trace
```python
def trace_expert_access(model):
    """Log all expert accesses for analysis"""
    access_log = []

    original_get = model.cache.get_expert
    def traced_get(layer_idx, expert_idx):
        access_log.append({
            'timestamp': time.time(),
            'layer': layer_idx,
            'expert': expert_idx,
            'hit': (layer_idx, expert_idx) in model.cache.cache
        })
        return original_get(layer_idx, expert_idx)

    model.cache.get_expert = traced_get
    return access_log
```

### 2. Memory Profiler
```python
def profile_memory_usage(model, input_data):
    """Profile memory usage during inference"""
    torch.cuda.reset_peak_memory_stats()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True
    ) as prof:
        output = model(input_data)

    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    return prof
```

## Best Practices

1. **Always use safetensors format** for secure model loading
2. **Monitor cache hit rates** - should be >60% in production
3. **Set memory limits** to prevent OOM
4. **Use mixed precision** (BF16) for 2× memory savings
5. **Enable async I/O** for better cache miss handling
6. **Profile before optimizing** - measure twice, cut once

---

*For optimization details, see [12_OPTIMIZATIONS_COMPLETE.md](12_OPTIMIZATIONS_COMPLETE.md)*
*For usage examples, see [21_OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)*