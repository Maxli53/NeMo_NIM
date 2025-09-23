#!/usr/bin/env python3
"""
LRU Cache for Expert Weights with Real Loading
Manages memory efficiently by caching frequently used experts
"""

import torch
from safetensors import safe_open
from collections import OrderedDict
from typing import Dict, Optional, Tuple
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpertLRUCache:
    """
    LRU cache for expert weights with actual safetensors loading
    """

    def __init__(self, model_path: str, max_size_gb: float = 5.0):
        self.model_path = Path(model_path)
        self.max_bytes = int(max_size_gb * 1e9)
        self.cache = OrderedDict()
        self.current_bytes = 0

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_load_time = 0

        # Shard mapping
        self.shards = {
            0: self.model_path / "model-00000-of-00002.safetensors",
            1: self.model_path / "model-00001-of-00002.safetensors",
        }

    def get_expert(self, layer_idx: int, expert_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get expert from cache or load from disk

        Args:
            layer_idx: Layer index (0-23)
            expert_idx: Expert index (0-31)

        Returns:
            Expert weights dict or None if not found
        """
        key = f"L{layer_idx}_E{expert_idx}"

        # Check cache
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)  # Mark as recently used
            logger.debug(f"Cache hit: {key} (hit rate: {self.get_hit_rate():.1%})")
            return self.cache[key]

        # Cache miss - load from disk
        self.misses += 1
        logger.debug(f"Cache miss: {key}, loading from disk...")

        start = time.time()
        expert_data = self._load_expert_from_disk(layer_idx, expert_idx)
        load_time = time.time() - start
        self.total_load_time += load_time

        if expert_data:
            self._add_to_cache(key, expert_data)
            logger.debug(f"Loaded {key} in {load_time*1000:.1f}ms")

        return expert_data

    def _load_expert_from_disk(self, layer_idx: int, expert_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single expert from safetensors file

        Args:
            layer_idx: Layer index
            expert_idx: Expert index

        Returns:
            Dict with expert weights
        """
        shard_idx = 0 if layer_idx < 12 else 1
        expert_data = {}

        try:
            with safe_open(self.shards[shard_idx], framework="pt", device="cpu") as f:
                # Gate-up projection weights
                gate_up_blocks_key = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj_blocks"
                gate_up_scales_key = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj_scales"
                gate_up_bias_key = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj_bias"

                if gate_up_blocks_key in f.keys():
                    # Load full tensor and slice expert
                    all_blocks = f.get_tensor(gate_up_blocks_key)
                    expert_data["gate_up_blocks"] = all_blocks[expert_idx].cuda()

                    all_scales = f.get_tensor(gate_up_scales_key)
                    expert_data["gate_up_scales"] = all_scales[expert_idx].cuda()

                    if gate_up_bias_key in f.keys():
                        all_bias = f.get_tensor(gate_up_bias_key)
                        expert_data["gate_up_bias"] = all_bias[expert_idx].cuda()

                # Down projection weights
                down_blocks_key = f"model.layers.{layer_idx}.mlp.experts.down_proj_blocks"
                down_scales_key = f"model.layers.{layer_idx}.mlp.experts.down_proj_scales"
                down_bias_key = f"model.layers.{layer_idx}.mlp.experts.down_proj_bias"

                if down_blocks_key in f.keys():
                    all_blocks = f.get_tensor(down_blocks_key)
                    expert_data["down_blocks"] = all_blocks[expert_idx].cuda()

                    all_scales = f.get_tensor(down_scales_key)
                    expert_data["down_scales"] = all_scales[expert_idx].cuda()

                    if down_bias_key in f.keys():
                        all_bias = f.get_tensor(down_bias_key)
                        expert_data["down_bias"] = all_bias[expert_idx].cuda()

        except Exception as e:
            logger.error(f"Error loading expert L{layer_idx}_E{expert_idx}: {e}")
            return {}

        return expert_data

    def _add_to_cache(self, key: str, expert_data: Dict[str, torch.Tensor]):
        """
        Add expert to cache with eviction if needed

        Args:
            key: Cache key
            expert_data: Expert weights to cache
        """
        # Calculate size
        size = sum(
            t.numel() * t.element_size()
            for t in expert_data.values()
            if isinstance(t, torch.Tensor)
        )

        # Evict if needed
        while self.current_bytes + size > self.max_bytes and self.cache:
            evicted_key = next(iter(self.cache))
            evicted_data = self.cache.pop(evicted_key)
            evicted_size = sum(
                t.numel() * t.element_size()
                for t in evicted_data.values()
                if isinstance(t, torch.Tensor)
            )
            self.current_bytes -= evicted_size
            self.evictions += 1
            logger.debug(f"Evicted {evicted_key} ({evicted_size/1e6:.1f}MB)")

        # Add to cache
        self.cache[key] = expert_data
        self.current_bytes += size

    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "hit_rate": self.get_hit_rate(),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "cache_size_gb": self.current_bytes / 1e9,
            "num_cached": len(self.cache),
            "avg_load_time_ms": (self.total_load_time / max(1, self.misses)) * 1000
        }

    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.current_bytes = 0
        logger.info("Cache cleared")


def test_expert_cache():
    """Test the expert LRU cache with actual loading"""
    # Use Windows path for Windows execution
    model_path = "C:/Users/maxli/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"

    logger.info("=" * 60)
    logger.info("EXPERT LRU CACHE TEST")
    logger.info("=" * 60)

    # Create cache with 1GB limit for testing
    cache = ExpertLRUCache(model_path, max_size_gb=1.0)

    logger.info("\n1. Testing cache with real expert loading...")

    # Test loading different experts
    test_cases = [
        (0, 0),  # Layer 0, Expert 0
        (0, 1),  # Layer 0, Expert 1
        (0, 0),  # Layer 0, Expert 0 (should hit cache)
        (1, 5),  # Layer 1, Expert 5
        (0, 0),  # Layer 0, Expert 0 (should hit cache)
    ]

    for layer_idx, expert_idx in test_cases:
        logger.info(f"\nLoading L{layer_idx}_E{expert_idx}...")
        start = time.time()
        expert = cache.get_expert(layer_idx, expert_idx)
        elapsed = time.time() - start

        if expert:
            num_params = sum(t.numel() for t in expert.values() if isinstance(t, torch.Tensor))
            size_mb = sum(t.numel() * t.element_size() for t in expert.values() if isinstance(t, torch.Tensor)) / 1e6
            logger.info(f"  Loaded in {elapsed*1000:.1f}ms")
            logger.info(f"  Parameters: {num_params:,}")
            logger.info(f"  Size: {size_mb:.1f}MB")
        else:
            logger.info(f"  Failed to load!")

    # Print statistics
    logger.info("\n2. Cache Statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        if isinstance(value, float) and key.endswith("rate"):
            logger.info(f"   {key}: {value:.1%}")
        elif isinstance(value, float):
            logger.info(f"   {key}: {value:.2f}")
        else:
            logger.info(f"   {key}: {value}")

    # Memory usage
    if torch.cuda.is_available():
        logger.info(f"\n3. GPU Memory:")
        logger.info(f"   Allocated: {torch.cuda.memory_allocated()/1e9:.3f} GB")

    logger.info("\n✅ Expert cache test complete!")
    return cache


if __name__ == "__main__":
    test_expert_cache()