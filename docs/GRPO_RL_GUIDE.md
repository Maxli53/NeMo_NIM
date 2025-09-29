# GPT-OSS GRPO/RL Training Guide

## Executive Summary

Unsloth now supports Reinforcement Learning (RL) with GRPO for GPT-OSS models, offering:
- **3x faster inference** (~21 tokens/s for 4-bit, ~30 tokens/s for BF16)
- **50% less VRAM usage** compared to other implementations
- **8x longer context** support
- **15GB VRAM** requirement for GPT-OSS-20B (fits on RTX 3090!)

## 🎯 What is GRPO?

**GRPO (Generative Reward-Prompted Optimization)** is a reinforcement learning technique that:
1. Generates candidate solutions
2. Evaluates them with reward functions
3. Optimizes the model to generate higher-reward outputs
4. Prevents reward hacking through careful design

## 🚀 Key Innovations

### 1. Inference Optimization
- **Custom inference engine** (vLLM doesn't support RL for GPT-OSS)
- **Unsloth Flex Attention** for O(N) memory usage (vs O(N²))
- **torch.compile optimizations** with combo kernels
- **4-bit RL support** (only framework supporting this)

### 2. Memory Optimization
- **Embedding offloading**: Reduces VRAM by 1GB via `offload_embeddings`
- **Weight sharing**: 50% reduction when vLLM becomes compatible
- **Gradient checkpointing**: Unsloth's custom implementation

### 3. Attention Handling
- **Attention sinks support** with differentiable backward pass
- **Left-padded masking** for batch generation
- **Dynamic KV cache** management

## ⚠️ Critical Issues to Avoid

### Flash Attention 3 Problem
**DO NOT USE FA3 for GPT-OSS RL!** It causes incorrect training losses because:
- FA3 doesn't support backward pass for attention sinks
- Many frameworks enable FA3 by default
- Must use Unsloth Flex Attention instead

### Memory Scaling
Without proper attention:
- Naive attention: O(N²) memory (unusable for long context)
- Flash Attention 3: Broken for GPT-OSS
- **Unsloth Flex Attention: O(N) memory** ✅

## 🛠️ Implementation Details

### Model Configuration
```python
from unsloth import FastLanguageModel, GRPOTrainer, GRPOConfig

# Load model with RL optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b",
    max_seq_length=768,  # Reduced for RL
    dtype=None,
    load_in_4bit=True,
    offload_embeddings=True  # NEW: Saves 1GB VRAM
)

# LoRA configuration for RL
model = FastLanguageModel.get_peft_model(
    model,
    r=4,  # Lower rank for RL
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=4,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

### GRPO Training Configuration
```python
training_args = GRPOConfig(
    # Generation settings
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    num_generations=2,  # Generate 2 candidates per prompt

    # Training settings
    learning_rate=5e-5,  # Lower than standard fine-tuning
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",

    # Batch settings
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    # Duration
    max_steps=100,

    # Logging
    logging_steps=1,
    output_dir="./grpo_outputs",

    # RL specific
    reward_model=None,  # Use custom reward functions
    max_new_tokens=512,
)
```

### Reward Functions

```python
def create_reward_functions():
    """Create reward functions for GRPO training"""

    def function_works(generated_code):
        """Check if generated code is valid Python"""
        try:
            exec(generated_code, {}, {})
            return 1.0
        except:
            return 0.0

    def no_cheating(generated_code):
        """Prevent importing unauthorized modules"""
        banned = ["numpy", "torch", "scipy", "numba"]
        for module in banned:
            if f"import {module}" in generated_code:
                return 0.0
        return 1.0

    def correctness_check(generated_code, test_inputs, expected_outputs):
        """Verify output correctness"""
        try:
            # Execute in sandboxed environment
            namespace = {}
            exec(generated_code, namespace)
            func = namespace.get('matmul')

            for inp, expected in zip(test_inputs, expected_outputs):
                result = func(inp['A'], inp['B'])
                if not np.allclose(result, expected):
                    return 0.0
            return 1.0
        except:
            return 0.0

    def speed_check(generated_code, baseline_time):
        """Benchmark performance"""
        try:
            # Time the generated function
            namespace = {}
            exec(generated_code, namespace)
            func = namespace.get('matmul')

            # Benchmark
            start = time.time()
            for _ in range(100):
                func(test_matrix_a, test_matrix_b)
            elapsed = time.time() - start

            # Reward if faster than baseline
            speedup = baseline_time / elapsed
            return max(0, min(2, speedup))  # Cap at 2x reward
        except:
            return 0.0

    return [function_works, no_cheating, correctness_check, speed_check]
```

## 🔒 Preventing Reward Hacking

### Common Hacking Strategies & Solutions

1. **Laziness** (Using optimized libraries)
   ```python
   # Solution: Ban imports
   if any(lib in code for lib in ["numpy", "torch", "scipy"]):
       return 0  # Zero reward
   ```

2. **Caching** (Storing results)
   ```python
   # Solution: Clear cache between calls
   import gc
   gc.collect()
   torch.cuda.empty_cache() if torch.cuda.is_available() else None
   ```

3. **Cheating** (Modifying timing function)
   ```python
   # Solution: Locked execution environment
   import types
   func = types.FunctionType(
       compiled_code.co_consts[0],
       {},  # Empty globals
       "matmul"
   )
   ```

4. **Global Variable Access**
   ```python
   # Solution: Restrict namespace
   exec(code, {"__builtins__": {}}, local_namespace)
   ```

## 📊 Training Process

### Complete GRPO Training Loop
```python
from unsloth import GRPOTrainer

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    reward_functions=create_reward_functions(),
)

# Custom training loop with monitoring
for step in range(training_args.max_steps):
    # Generate candidates
    prompts = dataset[step % len(dataset)]["prompt"]
    generations = trainer.generate(prompts)

    # Calculate rewards
    rewards = []
    for gen in generations:
        reward = sum(rf(gen) for rf in trainer.reward_functions)
        rewards.append(reward)

    # Update model
    loss = trainer.train_step(generations, rewards)

    # Logging
    if step % 10 == 0:
        print(f"Step {step}: Loss={loss:.4f}, Avg Reward={np.mean(rewards):.2f}")
```

## 🎯 Use Cases

### 1. Code Generation
- Train models to write optimized code
- Reward: Correctness + Speed + No cheating

### 2. Mathematical Reasoning
- Solve complex problems step-by-step
- Reward: Correct answer + Clear reasoning

### 3. Creative Writing
- Generate engaging content
- Reward: Human preference scores

### 4. Task Planning
- Break down complex tasks
- Reward: Completeness + Feasibility

## 🚦 Performance Benchmarks

| Configuration | Inference Speed | VRAM Usage | Context Length |
|--------------|----------------|------------|----------------|
| 4-bit RL | 21 tokens/s | 15GB | 8K tokens |
| BF16 RL | 30 tokens/s | 30GB | 8K tokens |
| Without Unsloth | 5 tokens/s | 30GB+ | 1K tokens |

## ⚙️ Technical Details

### Flex Attention Implementation
```python
# Attention with sinks (simplified)
def attention_with_sinks(Q, K, V, sinks):
    # Standard attention
    scores = Q @ K.T / sqrt(d)
    attn_weights = softmax(scores)
    output = attn_weights @ V

    # Apply sinks
    lse = logsumexp(scores)
    sink_mask = sigmoid(lse - sinks)
    output = output * sink_mask

    return output
```

### Left-Padded Masking
```python
# Handle variable-length sequences in batch
def create_batch_mask(seq_lengths, max_length):
    mask = torch.zeros(batch_size, max_length, max_length)
    for i, length in enumerate(seq_lengths):
        # Left padding means actual content starts at (max_length - length)
        start_idx = max_length - length
        mask[i, :, start_idx:] = 1
    return mask
```

## 📝 Best Practices

1. **Start with small LoRA rank** (r=4) for RL
2. **Use lower learning rates** (5e-5 vs 2e-4 for standard)
3. **Implement multiple reward functions** to prevent hacking
4. **Monitor reward distribution** to detect exploitation
5. **Use embedding offloading** to save 1GB VRAM
6. **Disable Flash Attention 3** explicitly
7. **Test reward functions** thoroughly before training

## 🔮 Future Developments

- **vLLM Integration**: Once RL support is added
- **Weight Sharing**: Additional 50% VRAM reduction
- **GSPO Support**: Alternative RL algorithm
- **Multi-GPU RL**: Distributed training

## 📚 Resources

- [GRPO Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb)
- [Unsloth RL Docs](https://docs.unsloth.ai/rl)
- [Reward Hacking Paper](https://arxiv.org/abs/reward-hacking)

## 🎉 Summary

GRPO/RL for GPT-OSS is now accessible on consumer hardware (RTX 3090) thanks to Unsloth's optimizations. The key is understanding:
- Proper attention handling (Flex, not FA3)
- Reward function design
- Preventing reward hacking
- Memory optimization techniques

With 15GB VRAM, you can now train frontier models with RL - previously only possible in well-funded labs!