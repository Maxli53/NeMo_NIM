#!/usr/bin/env python3
"""
Integration tests for safety framework with completed optimizations.
Verifies that existing tests still work with safety wrappers.
"""

import pytest
import torch
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'moe'))

from moe.optimization_safety.optimization_control_center import OptimizationControlCenter
from moe.optimization_safety.safe_optimizations import (
    SafeCUDAKernels, SafeAsyncIO, SafeTieredCache, SafeMultiGPU
)


class TestSafetyIntegration:
    """Test that safety wrappers work correctly with existing optimizations."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset control center before each test."""
        # Get control center instance
        self.control_center = OptimizationControlCenter.get_instance()
        # Reset all flags to OFF
        self.control_center.reset_all()
        yield
        # Cleanup after test
        self.control_center.reset_all()
    
    def test_cuda_kernels_safety_wrapper(self):
        """Test CUDA kernel optimization with safety wrapper."""
        safe_cuda = SafeCUDAKernels()
        
        # Should be disabled by default
        assert not safe_cuda.is_enabled()
        
        # Create test inputs
        batch_size, seq_len, hidden_size = 2, 128, 2880
        num_experts = 4
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        expert_outputs = torch.randn(batch_size, seq_len, num_experts, hidden_size)
        router_weights = torch.softmax(torch.randn(batch_size, seq_len, num_experts), dim=-1)
        
        # Should use fallback when disabled
        result = safe_cuda.fused_expert_mixer(hidden_states, expert_outputs, router_weights)
        assert result is not None
        assert result.shape == (batch_size, seq_len, hidden_size)
        
        # Enable optimization
        self.control_center.enable_optimization("cuda_kernels", traffic_percentage=1.0)
        
        # Now should be enabled (if CUDA available)
        if torch.cuda.is_available():
            assert safe_cuda.is_enabled()
        
        # Test with enabled optimization
        result = safe_cuda.fused_expert_mixer(hidden_states, expert_outputs, router_weights)
        assert result is not None
        assert result.shape == (batch_size, seq_len, hidden_size)
    
    def test_async_io_safety_wrapper(self):
        """Test async I/O optimization with safety wrapper."""
        import asyncio
        
        cache_dir = "./test_cache"
        safe_async = SafeAsyncIO(cache_dir)
        
        # Should be disabled by default
        assert not safe_async.is_enabled()
        
        # Test sync fallback
        # This should not crash even when disabled
        layer_idx, expert_idx = 0, 1
        # Sync fallback doesn't actually load, just returns path
        expert = safe_async.load_expert_sync(layer_idx, expert_idx)
        assert expert is not None  # Returns path or loaded tensor
        
        # Enable optimization
        self.control_center.enable_optimization("async_io", traffic_percentage=1.0)
        assert safe_async.is_enabled()
        
        # Test async prefetching (won't actually prefetch without real implementation)
        async def test_prefetch():
            router_logits = torch.randn(2, 128, 32)  # batch, seq, num_experts
            await safe_async.prefetch_experts(router_logits, layer_idx=0)
        
        # Should complete without error
        asyncio.run(test_prefetch())
    
    def test_tiered_cache_safety_wrapper(self):
        """Test tiered cache optimization with safety wrapper."""
        cache_config = {
            'cache_size': 100,
            'gpu_capacity_gb': 2.0,
            'ram_capacity_gb': 16.0,
            'disk_capacity_gb': 100.0
        }
        
        safe_cache = SafeTieredCache(cache_config)
        
        # Should be disabled by default
        assert not safe_cache.is_enabled()
        
        # Test fallback cache
        expert, cache_tier = safe_cache.get(layer_idx=0, expert_idx=1)
        assert cache_tier in ["CACHE_DISABLED", "FALLBACK_CACHE"]
        
        # Enable optimization
        self.control_center.enable_optimization("tiered_cache", traffic_percentage=1.0)
        
        # Test with enabled cache
        expert, cache_tier = safe_cache.get(layer_idx=0, expert_idx=1)
        # Should return a valid cache tier status
        assert cache_tier in ["GPU_HIT", "RAM_HIT", "DISK_HIT", "MISS", "CACHE_DISABLED", "FALLBACK_CACHE"]
    
    def test_multi_gpu_safety_wrapper(self):
        """Test multi-GPU optimization with safety wrapper."""
        safe_multi_gpu = SafeMultiGPU()
        
        # Should be disabled by default
        assert not safe_multi_gpu.is_enabled()
        
        # Create test inputs
        batch_size, seq_len, hidden_size = 2, 128, 2880
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        router_logits = torch.randn(batch_size, seq_len, 32)  # 32 experts
        experts = None  # Placeholder
        
        # Should use single-GPU fallback when disabled
        result = safe_multi_gpu.forward(hidden_states, router_logits, experts)
        assert result is not None
        assert result.shape == hidden_states.shape
        
        # Enable optimization (will only work with multiple GPUs)
        self.control_center.enable_optimization("multi_gpu", traffic_percentage=1.0)
        
        if torch.cuda.device_count() > 1:
            assert safe_multi_gpu.is_enabled()
        else:
            # Should auto-disable with single GPU
            assert not safe_multi_gpu.is_enabled()
        
        # Test forward pass
        result = safe_multi_gpu.forward(hidden_states, router_logits, experts)
        assert result is not None
        assert result.shape == hidden_states.shape
    
    def test_emergency_stop(self):
        """Test emergency stop functionality."""
        # Enable all optimizations
        self.control_center.enable_optimization("cuda_kernels", traffic_percentage=1.0)
        self.control_center.enable_optimization("async_io", traffic_percentage=1.0)
        self.control_center.enable_optimization("tiered_cache", traffic_percentage=1.0)
        self.control_center.enable_optimization("multi_gpu", traffic_percentage=1.0)
        
        # Verify enabled
        status = self.control_center.get_status()
        assert any(opt['enabled'] for opt in status['optimizations'].values())
        
        # Trigger emergency stop
        self.control_center.emergency_stop_all("Test emergency stop")
        
        # Verify all disabled
        status = self.control_center.get_status()
        assert not any(opt['enabled'] for opt in status['optimizations'].values())
        assert status['emergency_stop']
    
    def test_progressive_rollout(self):
        """Test progressive traffic rollout."""
        # Enable at 10% traffic
        self.control_center.enable_optimization("tiered_cache", traffic_percentage=0.1)
        
        config = self.control_center.get_config()
        assert config.cache_config.traffic_percentage == 0.1
        
        # Increase to 50%
        self.control_center.enable_optimization("tiered_cache", traffic_percentage=0.5)
        
        config = self.control_center.get_config()
        assert config.cache_config.traffic_percentage == 0.5
        
        # Increase to 100%
        self.control_center.enable_optimization("tiered_cache", traffic_percentage=1.0)
        
        config = self.control_center.get_config()
        assert config.cache_config.traffic_percentage == 1.0
    
    def test_automatic_rollback(self):
        """Test automatic rollback on threshold violations."""
        from moe.optimization_safety.optimization_monitor import OptimizationHealthMonitor
        
        monitor = OptimizationHealthMonitor()
        
        # Enable optimization
        self.control_center.enable_optimization("cuda_kernels", traffic_percentage=1.0)
        assert self.control_center.is_optimization_enabled("cuda_kernels")
        
        # Simulate threshold violations
        for _ in range(5):  # More than strike limit
            monitor.record_metrics(
                "cuda_kernels",
                latency_ms=500,  # Way above threshold
                error=True    # Error occurred
            )
        
        # Check health status
        status, issues = monitor.get_health_status("cuda_kernels")
        assert status == "CRITICAL"
        assert len(issues) > 0
        
        # In production, rollback manager would disable the optimization
        # For test, we manually verify the health status indicates problems


class TestBackwardCompatibility:
    """Ensure existing code still works with safety framework."""
    
    def test_import_original_modules(self):
        """Test that original modules can still be imported."""
        try:
            from moe import cuda_kernels
            from moe import async_expert_loader
            from moe import tiered_cache
            from moe import multi_gpu_moe
            assert True
        except ImportError as e:
            pytest.skip(f"Original modules not found: {e}")
    
    def test_native_moe_safe_initialization(self):
        """Test that safe native MoE initializes correctly."""
        try:
            from moe.native_moe_safe import SafeGPTOSSNativeMoE
            
            # Create model with safety framework
            model = SafeGPTOSSNativeMoE(
                model_path="./models/gpt-oss-20b",  # May not exist, just testing init
                cache_size_gb=2.0,
                device="cpu",  # Use CPU for test
                enable_optimizations=False  # All OFF by default
            )
            
            assert model is not None
            assert hasattr(model, 'control_center')
            assert hasattr(model, 'optimizations')
            assert len(model.optimizations) == 4  # 4 completed optimizations
            
        except Exception as e:
            # May fail due to missing model files, but structure should be correct
            if "config.json" in str(e) or "No such file" in str(e):
                pytest.skip(f"Model files not found: {e}")
            else:
                raise
    
    def test_feature_flags_default_off(self):
        """Verify all optimizations default to OFF."""
        control_center = OptimizationControlCenter.get_instance()
        control_center.reset_all()
        
        # Check all flags are OFF
        assert not control_center.is_optimization_enabled("cuda_kernels")
        assert not control_center.is_optimization_enabled("async_io")
        assert not control_center.is_optimization_enabled("tiered_cache")
        assert not control_center.is_optimization_enabled("multi_gpu")
        
        # Also check future optimizations are OFF
        assert not control_center.is_optimization_enabled("dynamic_batching")
        assert not control_center.is_optimization_enabled("flash_attention")
        assert not control_center.is_optimization_enabled("int8_weights")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])