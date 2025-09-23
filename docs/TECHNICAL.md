# Technical Architecture

## Overview

Native PyTorch implementation of a 20B parameter Mixture of Experts (MoE) model optimized for single-GPU inference on RTX 3090 (24GB VRAM).

## Core Architecture

### MoE Components

```python
class MoELayer:
    """Core MoE implementation with top-k routing"""

    Components:
    - Router: Selects top-k experts per token
    - Experts: 32 FFN experts per layer
    - Mixer: Combines expert outputs
    - Cache: LRU cache for expert weights
```

### Key Design Decisions

1. **Top-k=4 Expert Selection**
   - Only 4 out of 32 experts active per token
   - 87.5% memory reduction (32/4 = 8x theoretical)
   - Maintains quality with sparse activation

2. **Native PyTorch Implementation**
   - No dependency on DeepSpeed/FairScale
   - Direct control over memory management
   - Compatible with standard PyTorch optimizations

3. **Safetensors Weight Loading**
   - Efficient memory-mapped loading
   - Selective expert loading
   - Zero-copy where possible

## Implementation Details

### 1. Expert Routing

```python
def route_tokens(self, inputs):
    """Route tokens to top-k experts"""
    # Shape: [batch, seq_len, hidden_dim]
    router_logits = self.router(inputs)  # [B, S, num_experts]

    # Select top-k experts
    top_k_logits, top_k_indices = torch.topk(
        router_logits, k=self.top_k, dim=-1
    )

    # Normalize weights
    top_k_weights = F.softmax(top_k_logits, dim=-1)

    return top_k_indices, top_k_weights
```

### 2. Expert Mixing

```python
def mix_expert_outputs(self, expert_outputs, weights):
    """Combine outputs from selected experts"""
    # expert_outputs: [B, S, K, H]
    # weights: [B, S, K]

    # Weighted sum
    mixed = torch.einsum('bskh,bsk->bsh', expert_outputs, weights)
    return mixed
```

### 3. Memory Management

```python
class ExpertCache:
    """LRU cache for expert weights"""

    def __init__(self, max_memory_gb=5.0):
        self.cache = OrderedDict()
        self.max_memory = max_memory_gb * 1e9

    def get_expert(self, layer_idx, expert_idx):
        key = f"L{layer_idx}_E{expert_idx}"
        if key in self.cache:
            self.cache.move_to_end(key)  # LRU
            return self.cache[key]
        return self.load_expert(layer_idx, expert_idx)
```

### 4. Attention Optimization

```python
# Use SDPA for Flash Attention
from torch.nn.functional import scaled_dot_product_attention

def forward_attention(self, q, k, v):
    """Optimized attention using SDPA"""
    # Shapes: [B, H, S, D]
    attn_output = scaled_dot_product_attention(
        q, k, v,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0/math.sqrt(self.head_dim)
    )
    return attn_output
```

## File Structure

```
src/moe/
├── native_moe_loader_v2.py      # Model loading with top-k
├── expert_cache.py               # LRU expert caching
├── async_expert_loader.py        # Async I/O for weights
├── optimization_safety/          # Safety framework
│   ├── optimization_control_center.py
│   ├── optimization_monitor.py
│   └── rollback_manager.py
└── extensions/
    ├── flash_attention.py        # SDPA wrapper
    └── quantization.py           # INT8 support (WIP)
```

## Memory Layout

### Full Model (All Experts)
```
32 experts × 24 layers = 768 expert FFNs
Memory per expert: ~50MB
Total: 768 × 50MB = 38.4GB (exceeds 24GB VRAM)
```

### Top-k=4 Configuration
```
4 experts × 24 layers = 96 active expert FFNs
Memory per expert: ~50MB
Total: 96 × 50MB = 4.8GB
Non-expert weights: ~2.5GB
Total: ~7.3GB (fits in 24GB VRAM)
```

## Safety Framework

### Feature Flags
```python
@dataclass
class OptimizationFlags:
    fp16: bool = True           # ✅ Enabled
    sdpa: bool = True           # ✅ Enabled
    top_k: int = 4              # ✅ Enabled
    torch_compile: bool = False # ❌ Disabled (regression)
    int8: bool = False          # ❌ Disabled (too slow)
    mixed_precision: bool = False # ❌ Disabled (overhead)
```

### Monitoring
```python
class HealthMonitor:
    """Real-time performance monitoring"""

    thresholds = {
        'latency_ms': 500,      # Max first token
        'throughput_tps': 6,    # Min tokens/sec
        'memory_gb': 22,        # Max VRAM
        'accuracy': 0.95        # Min quality
    }
```

### Rollback System
```python
class RollbackManager:
    """Automatic rollback on failure"""

    def check_health(self, metrics):
        if metrics['latency_ms'] > self.thresholds['latency_ms']:
            self.rollback_optimization('sdpa')
        if metrics['memory_gb'] > self.thresholds['memory_gb']:
            self.reduce_batch_size()
```

## Integration Points

### 1. Model Loading
```python
from src.moe.native_moe_loader_v2 import MoEModelLoader

loader = MoEModelLoader("gpt-oss-20b/original")
model = loader.create_model_fp16(top_k=4, full_layers=True)
```

### 2. Inference
```python
# Production configuration
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=128,
        temperature=0.7
    )
```

### 3. Optimization Control
```python
from src.moe.optimization_safety import get_control_center

center = get_control_center()
center.enable_optimization("sdpa", validate=True)
center.disable_optimization("torch_compile")  # Known regression
```

## Known Limitations

1. **INT8 Quantization**
   - dtype mismatch between FP16/INT8 layers
   - 5x performance degradation when working
   - Needs proper input casting fix

2. **Batch Processing**
   - Only tested with batch_size=1
   - Mixed precision might help with larger batches

3. **Sequence Length**
   - Tested up to 128 tokens
   - Longer sequences not validated

4. **Model Weights**
   - Currently using random initialization
   - Need actual pretrained weights for quality metrics

## Future Improvements

### High Priority
- Fix INT8 dtype handling
- Test batch_size > 1
- Load pretrained weights

### Medium Priority
- Dynamic expert routing
- Sequence length >512
- Multi-GPU support

### Low Priority
- Kernel fusion optimizations
- Custom CUDA kernels
- Speculative decoding