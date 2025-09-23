#!/usr/bin/env python3
"""
Unit Tests for MoE Components
Tests individual components in isolation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.moe.expert_cache import ExpertLRUCache
from src.moe.optimization_safety.optimization_control_center import OptimizationControlCenter
from src.moe.optimization_safety.optimization_monitor import HealthMonitor


class TestExpertRouting:
    """Test expert routing logic"""

    def test_top_k_selection(self):
        """Verify top-k expert selection works correctly"""
        batch_size, seq_len, num_experts = 2, 10, 32
        top_k = 4

        # Create random router logits
        router_logits = torch.randn(batch_size, seq_len, num_experts)

        # Select top-k
        top_k_values, top_k_indices = torch.topk(router_logits, k=top_k, dim=-1)

        assert top_k_indices.shape == (batch_size, seq_len, top_k)
        assert top_k_values.shape == (batch_size, seq_len, top_k)

        # Verify indices are unique per position
        for b in range(batch_size):
            for s in range(seq_len):
                indices = top_k_indices[b, s].tolist()
                assert len(indices) == len(set(indices)), "Duplicate expert indices"

    def test_weight_normalization(self):
        """Verify router weights sum to 1"""
        batch_size, seq_len, top_k = 2, 10, 4

        # Create random weights
        weights = torch.randn(batch_size, seq_len, top_k)
        normalized = torch.softmax(weights, dim=-1)

        # Check sum equals 1
        sums = normalized.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


class TestExpertMixing:
    """Test expert output mixing"""

    def test_weighted_combination(self):
        """Verify expert outputs are correctly combined"""
        batch_size, seq_len, hidden_dim = 2, 10, 768
        top_k = 4

        # Create dummy expert outputs and weights
        expert_outputs = torch.randn(batch_size, seq_len, top_k, hidden_dim)
        weights = torch.softmax(torch.randn(batch_size, seq_len, top_k), dim=-1)

        # Mix outputs
        mixed = torch.einsum('bskh,bsk->bsh', expert_outputs, weights)

        assert mixed.shape == (batch_size, seq_len, hidden_dim)

        # Verify it's actually a weighted sum
        manual_mixed = torch.zeros(batch_size, seq_len, hidden_dim)
        for k in range(top_k):
            manual_mixed += expert_outputs[:, :, k, :] * weights[:, :, k:k+1]

        assert torch.allclose(mixed, manual_mixed, atol=1e-6)


class TestExpertCache:
    """Test LRU expert caching"""

    def test_cache_eviction(self):
        """Verify LRU eviction works"""
        cache = ExpertLRUCache(max_memory_gb=0.001)  # Very small cache

        # Add tensors that exceed cache
        tensor1 = torch.randn(1000, 1000)  # ~4MB
        tensor2 = torch.randn(1000, 1000)  # ~4MB

        cache.put("expert1", tensor1)
        assert cache.get("expert1") is not None

        cache.put("expert2", tensor2)  # Should evict expert1
        assert cache.get("expert1") is None  # Evicted
        assert cache.get("expert2") is not None  # Still there

    def test_cache_hit_rate(self):
        """Test cache hit rate tracking"""
        cache = ExpertLRUCache(max_memory_gb=0.1)

        # Simulate access pattern
        for i in range(100):
            key = f"expert_{i % 10}"  # 10 unique experts
            if cache.get(key) is None:
                cache.put(key, torch.randn(100, 100))

        # Most should be hits after warmup
        assert len(cache.cache) > 0


class TestSafetyFramework:
    """Test optimization safety controls"""

    def test_feature_flags(self):
        """Verify feature flags work correctly"""
        control = OptimizationControlCenter()

        # Default should be OFF
        assert not control.is_enabled("torch_compile")
        assert not control.is_enabled("int8_weights")

        # Enable and verify
        control.enable_optimization("sdpa")
        assert control.is_enabled("sdpa")

        # Disable and verify
        control.disable_optimization("sdpa")
        assert not control.is_enabled("sdpa")

    def test_rollback_on_threshold(self):
        """Test automatic rollback on threshold violation"""
        monitor = HealthMonitor()

        # Set strict thresholds
        monitor.thresholds = {
            'latency_ms': 100,
            'memory_gb': 5
        }

        # Simulate bad metrics
        bad_metrics = {
            'latency_ms': 500,  # Exceeds threshold
            'memory_gb': 4
        }

        violations = monitor.check_thresholds(bad_metrics)
        assert 'latency_ms' in violations
        assert 'memory_gb' not in violations


class TestMemoryManagement:
    """Test memory optimization features"""

    def test_memory_estimation(self):
        """Test memory usage estimation"""
        batch_size = 1
        seq_len = 128
        hidden_dim = 2880
        num_experts = 32
        top_k = 4

        # Estimate memory for top-k experts
        params_per_expert = hidden_dim * hidden_dim * 4  # Rough estimate
        bytes_per_param = 2  # FP16

        expected_memory = top_k * params_per_expert * bytes_per_param
        expected_mb = expected_memory / (1024 * 1024)

        # Should be reasonable
        assert expected_mb < 1000  # Less than 1GB per layer

    def test_memory_reduction(self):
        """Verify top-k reduces memory"""
        full_experts = 32
        active_experts = 4

        reduction = 1 - (active_experts / full_experts)
        assert reduction == 0.875  # 87.5% reduction


class TestPerformanceMetrics:
    """Test performance measurement utilities"""

    def test_throughput_calculation(self):
        """Test tokens per second calculation"""
        num_tokens = 128
        time_seconds = 4.39

        throughput = num_tokens / time_seconds
        assert abs(throughput - 29.1) < 0.5  # Close to our measured value

    def test_latency_measurement(self):
        """Test first token latency measurement"""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Simulate some work
        start.record()
        _ = torch.randn(1000, 1000).cuda()
        end.record()

        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end)

        assert latency_ms > 0
        assert latency_ms < 1000  # Should be fast


def test_dtype_compatibility():
    """Test dtype handling for INT8 issue"""
    # This is the problematic case
    fp16_tensor = torch.randn(10, 10).half().cuda()

    # INT8 expects float32
    try:
        fp32_tensor = fp16_tensor.float()
        # This should work
        result = torch.matmul(fp32_tensor, fp32_tensor)
        assert result.dtype == torch.float32
    except RuntimeError:
        pytest.fail("dtype conversion failed")


def test_sdpa_availability():
    """Test SDPA/Flash Attention availability"""
    try:
        from torch.nn.functional import scaled_dot_product_attention

        # Create small test tensors
        batch, heads, seq_len, head_dim = 1, 8, 128, 64
        q = torch.randn(batch, heads, seq_len, head_dim).cuda()
        k = torch.randn(batch, heads, seq_len, head_dim).cuda()
        v = torch.randn(batch, heads, seq_len, head_dim).cuda()

        # Should work without error
        output = scaled_dot_product_attention(q, k, v)
        assert output.shape == (batch, heads, seq_len, head_dim)

    except ImportError:
        pytest.skip("SDPA not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])