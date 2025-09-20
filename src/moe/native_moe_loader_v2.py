#!/usr/bin/env python3
"""
Native MoE Loader for GPT-OSS-20B
Implements dynamic expert loading with only top-4 experts per layer
"""

import os
import torch
import torch.nn as nn
from safetensors import safe_open
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import json
from pathlib import Path
import logging
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpertLRUCache:
    """LRU cache for keeping frequently used experts in memory"""

    def __init__(self, max_memory_gb: float = 5.0):
        self.max_memory_bytes = int(max_memory_gb * 1e9)
        self.cache = OrderedDict()
        self.current_memory = 0

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, tensor: torch.Tensor):
        tensor_bytes = tensor.numel() * tensor.element_size()

        # Evict old items if needed
        while self.current_memory + tensor_bytes > self.max_memory_bytes and self.cache:
            evicted_key, evicted_tensor = self.cache.popitem(last=False)
            evicted_bytes = evicted_tensor.numel() * evicted_tensor.element_size()
            self.current_memory -= evicted_bytes
            logger.debug(f"Evicted {evicted_key}, freed {evicted_bytes/1e6:.1f}MB")

        self.cache[key] = tensor
        self.current_memory += tensor_bytes

    def clear(self):
        self.cache.clear()
        self.current_memory = 0


class GPTOSSNativeMoE:
    """Native MoE implementation with dynamic expert loading"""

    def __init__(self, model_path: str, cache_size_gb: float = 5.0):
        self.model_path = Path(model_path)
        self.expert_cache = ExpertLRUCache(cache_size_gb)

        # Load config
        with open(self.model_path / "config.json") as f:
            self.config = json.load(f)

        self.num_layers = self.config.get("num_hidden_layers", 24)
        self.num_experts = self.config.get("num_local_experts", 32)
        self.experts_per_token = self.config.get("experts_per_token", 4)
        self.hidden_size = self.config.get("hidden_size", 2880)

        logger.info(f"Model config: {self.num_layers} layers, {self.num_experts} experts, top-{self.experts_per_token}")

        # Map safetensors shards
        self.shards = {
            0: self.model_path / "model-00000-of-00002.safetensors",  # Layers 0-11
            1: self.model_path / "model-00001-of-00002.safetensors",  # Layers 12-23
            2: self.model_path / "model-00002-of-00002.safetensors",  # Embeddings, LM head
        }

        # Load routers (small, keep in memory)
        self.routers = self._load_routers()
        logger.info(f"Loaded routers for {len(self.routers)} layers")

    def _load_routers(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Load all router weights (they're small)"""
        routers = {}

        for layer_idx in range(self.num_layers):
            shard_idx = 0 if layer_idx < 12 else 1

            with safe_open(self.shards[shard_idx], framework="pt", device="cpu") as f:
                router_weight_key = f"model.layers.{layer_idx}.mlp.router.weight"
                router_bias_key = f"model.layers.{layer_idx}.mlp.router.bias"

                if router_weight_key in f.keys():
                    routers[layer_idx] = {
                        "weight": f.get_tensor(router_weight_key).cuda(),
                        "bias": f.get_tensor(router_bias_key).cuda()
                    }

        return routers

    def route_tokens(self, hidden_states: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k experts for each token
        Returns: (expert_indices, expert_weights)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Get router for this layer
        router = self.routers[layer_idx]

        # Compute routing scores
        # hidden_states: [batch, seq, hidden] @ weight.T: [hidden, num_experts]
        scores = hidden_states @ router["weight"].T + router["bias"]  # [batch, seq, num_experts]

        # Select top-k experts
        expert_weights, expert_indices = torch.topk(scores, k=self.experts_per_token, dim=-1)
        expert_weights = torch.softmax(expert_weights, dim=-1)

        return expert_indices, expert_weights

    def load_experts(self, layer_idx: int, expert_indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        Load only the specified experts for a layer
        Returns dict with gate_up_proj and down_proj weights
        """
        experts = {}

        for expert_idx in expert_indices:
            cache_key = f"layer_{layer_idx}_expert_{expert_idx}"

            # Check cache first
            cached = self.expert_cache.get(cache_key)
            if cached is not None:
                experts[expert_idx] = cached
                continue

            # Load from disk
            shard_idx = 0 if layer_idx < 12 else 1

            with safe_open(self.shards[shard_idx], framework="pt", device="cpu") as f:
                # Load expert weights (MXFP4 format)
                expert_weights = {}

                # Gate up projection
                gate_up_blocks_key = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj_blocks"
                gate_up_scales_key = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj_scales"
                gate_up_bias_key = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj_bias"

                if gate_up_blocks_key in f.keys():
                    # Extract only this expert (first dim is expert index)
                    all_blocks = f.get_tensor(gate_up_blocks_key)
                    expert_weights["gate_up_blocks"] = all_blocks[expert_idx].cuda()

                    all_scales = f.get_tensor(gate_up_scales_key)
                    expert_weights["gate_up_scales"] = all_scales[expert_idx].cuda()

                    all_bias = f.get_tensor(gate_up_bias_key)
                    expert_weights["gate_up_bias"] = all_bias[expert_idx].cuda()

                # Down projection
                down_blocks_key = f"model.layers.{layer_idx}.mlp.experts.down_proj_blocks"
                down_scales_key = f"model.layers.{layer_idx}.mlp.experts.down_proj_scales"
                down_bias_key = f"model.layers.{layer_idx}.mlp.experts.down_proj_bias"

                if down_blocks_key in f.keys():
                    all_blocks = f.get_tensor(down_blocks_key)
                    expert_weights["down_blocks"] = all_blocks[expert_idx].cuda()

                    all_scales = f.get_tensor(down_scales_key)
                    expert_weights["down_scales"] = all_scales[expert_idx].cuda()

                    all_bias = f.get_tensor(down_bias_key)
                    expert_weights["down_bias"] = all_bias[expert_idx].cuda()

                # Cache the expert
                self.expert_cache.put(cache_key, expert_weights)
                experts[expert_idx] = expert_weights

        return experts

    def mxfp4_to_bfloat16(self, blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        Convert MXFP4 quantized weights to bfloat16
        This is a simplified version - full implementation would handle block-wise dequantization
        """
        # For now, just cast to bfloat16 (assumes pre-dequantized or fallback)
        # Real implementation would: blocks * scales with proper reshaping
        return blocks.to(torch.bfloat16)

    def compute_expert_outputs(
        self,
        hidden_states: torch.Tensor,
        experts: Dict[int, Dict[str, torch.Tensor]],
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted mixture of expert outputs
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        output = torch.zeros_like(hidden_states)

        # Process each token
        for b in range(batch_size):
            for s in range(seq_len):
                token_hidden = hidden_states[b, s]  # [hidden_dim]
                token_output = torch.zeros(hidden_dim, device=hidden_states.device)

                # Get this token's experts
                token_expert_indices = expert_indices[b, s]  # [k]
                token_expert_weights = expert_weights[b, s]  # [k]

                # Compute weighted sum of expert outputs
                for i, expert_idx in enumerate(token_expert_indices):
                    expert_idx = expert_idx.item()
                    weight = token_expert_weights[i]

                    if expert_idx in experts:
                        expert = experts[expert_idx]

                        # Simple FFN computation (would need proper MXFP4 dequantization)
                        # For now, assuming weights are already in usable format
                        # Real implementation would handle gate_up projection properly

                        # This is simplified - actual GPT-OSS uses SwiGLU activation
                        intermediate = token_hidden  # Placeholder
                        expert_output = intermediate * weight

                        token_output += expert_output

                output[b, s] = token_output

        return output

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        stats = {
            "cache_size_mb": self.expert_cache.current_memory / 1e6,
            "cache_items": len(self.expert_cache.cache),
            "ram_gb": psutil.Process().memory_info().rss / 1e9
        }

        if torch.cuda.is_available():
            stats["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            stats["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

        return stats

    def forward_layer(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Forward pass through one MoE layer with dynamic expert loading
        """
        # Route tokens to experts
        expert_indices, expert_weights = self.route_tokens(hidden_states, layer_idx)

        # Get unique experts needed for this batch
        unique_experts = torch.unique(expert_indices).cpu().tolist()
        logger.debug(f"Layer {layer_idx}: Loading experts {unique_experts}")

        # Load only needed experts
        experts = self.load_experts(layer_idx, unique_experts)

        # Compute expert outputs
        output = self.compute_expert_outputs(
            hidden_states, experts, expert_indices, expert_weights
        )

        return output


def test_native_loader():
    """Test the native MoE loader"""
    import time

    model_path = "C:/Users/maxli/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"

    logger.info("Initializing Native MoE Loader...")
    moe = GPTOSSNativeMoE(model_path, cache_size_gb=5.0)

    # Test routing
    logger.info("\nTesting token routing...")
    batch_size, seq_len = 1, 10
    hidden_states = torch.randn(batch_size, seq_len, moe.hidden_size).cuda()

    expert_indices, expert_weights = moe.route_tokens(hidden_states, layer_idx=0)
    logger.info(f"Routed to experts: {expert_indices[0, 0].cpu().tolist()}")
    logger.info(f"Expert weights: {expert_weights[0, 0].cpu().tolist()}")

    # Test expert loading
    logger.info("\nTesting expert loading...")
    start = time.time()
    experts = moe.load_experts(0, [0, 1, 2, 3])
    load_time = time.time() - start
    logger.info(f"Loaded 4 experts in {load_time:.3f} seconds")

    # Test forward pass
    logger.info("\nTesting forward pass through layer...")
    start = time.time()
    output = moe.forward_layer(hidden_states, layer_idx=0)
    forward_time = time.time() - start
    logger.info(f"Forward pass completed in {forward_time:.3f} seconds")
    logger.info(f"Output shape: {output.shape}")

    # Memory stats
    stats = moe.get_memory_stats()
    logger.info("\nMemory Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value:.2f}")

    logger.info("\n✅ Native MoE loader test complete!")

    return moe


if __name__ == "__main__":
    test_native_loader()