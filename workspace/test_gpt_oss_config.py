#!/usr/bin/env python3
"""
Test GPT-OSS configuration and MoE parameters
"""

from nemo.collections import llm

# Create GPT-OSS-20B configuration
config = llm.GPTOSSConfig20B()

print("GPT-OSS-20B Configuration:")
print(f"  Model: GPT-OSS-20B")
print(f"  Num Layers: {config.num_layers}")
print(f"  Num MoE Experts: {config.num_moe_experts}")
print(f"  MoE Router TopK: {config.moe_router_topk}")
print(f"  Hidden Size: {config.hidden_size}")
print(f"  Vocab Size: {config.vocab_size}")
print(f"  Attention Heads: {config.num_attention_heads}")
print(f"  Sequence Length: {config.seq_length}")

print("\nMoE Configuration Details:")
print(f"  Total experts: {config.num_moe_experts}")
print(f"  Active experts per token: {config.moe_router_topk}")
print(f"  Can be configured from 1 to {config.num_moe_experts}")

# Test modifying topk
print("\nTesting TopK modification:")
for topk in [1, 2, 4, 8, 16, 32]:
    config.moe_router_topk = topk
    active_params = config.hidden_size * topk  # Simplified calculation
    print(f"  topk={topk:2d}: ~{active_params/1e9:.1f}B active params per token")

print("\nConfiguration test successful!")