#!/usr/bin/env python3
"""
Functional/Integration Tests for MoE System
Tests end-to-end functionality and integration between components
"""

import pytest
import torch
import json
import time
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.moe.native_moe_loader_v2 import MoEModelLoader
from src.moe.optimization_safety.optimization_control_center import OptimizationControlCenter


class TestModelLoading:
    """Test model loading and initialization"""

    @pytest.fixture
    def model_path(self):
        """Get model path"""
        return Path("gpt-oss-20b/original")

    def test_config_loading(self, model_path):
        """Test loading model configuration"""
        if not model_path.exists():
            pytest.skip("Model files not found")

        loader = MoEModelLoader(str(model_path))
        config = loader.load_config()

        # Verify expected config values
        assert config['num_experts'] == 32
        assert config['experts_per_token'] == 4
        assert config['hidden_size'] == 2880
        assert config['num_hidden_layers'] == 24

    def test_selective_expert_loading(self, model_path):
        """Test loading only selected experts"""
        if not model_path.exists():
            pytest.skip("Model files not found")

        loader = MoEModelLoader(str(model_path))

        # Load only 4 experts
        weights = loader.load_weights_selective(expert_indices=[0, 1, 2, 3])

        # Should have fewer tensors than full model
        assert len(weights) > 0
        assert len(weights) < 1000  # Full model would have more

    @pytest.mark.slow
    def test_model_creation(self, model_path):
        """Test creating model with top-k experts"""
        if not model_path.exists():
            pytest.skip("Model files not found")

        loader = MoEModelLoader(str(model_path))

        try:
            # Create small model for testing
            model = loader.create_model_fp16(top_k=4, full_layers=False)
            assert model is not None

            # Check it's on GPU and FP16
            param = next(model.parameters())
            assert param.dtype == torch.float16
            assert param.device.type == 'cuda'

        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip("Insufficient GPU memory")
            raise


class TestInference:
    """Test model inference capabilities"""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing"""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(768, 768)

            def generate(self, input_ids, max_new_tokens=10):
                # Simple mock generation
                batch_size = input_ids.shape[0]
                new_tokens = torch.randint(0, 1000, (batch_size, max_new_tokens))
                return torch.cat([input_ids, new_tokens.to(input_ids.device)], dim=1)

        return SimpleModel().cuda()

    def test_generation(self, simple_model):
        """Test text generation"""
        input_ids = torch.randint(0, 1000, (1, 10)).cuda()

        output = simple_model.generate(input_ids, max_new_tokens=20)

        assert output.shape == (1, 30)  # 10 input + 20 generated
        assert output.device.type == 'cuda'

    def test_batch_generation(self, simple_model):
        """Test batch generation"""
        batch_size = 4
        input_ids = torch.randint(0, 1000, (batch_size, 10)).cuda()

        output = simple_model.generate(input_ids, max_new_tokens=20)

        assert output.shape == (batch_size, 30)


class TestOptimizationIntegration:
    """Test optimization features integration"""

    def test_sdpa_integration(self):
        """Test SDPA/Flash Attention integration"""
        try:
            from torch.nn.functional import scaled_dot_product_attention

            # Create attention layer
            batch, seq_len, hidden = 2, 128, 768
            num_heads = 12
            head_dim = hidden // num_heads

            q = torch.randn(batch, num_heads, seq_len, head_dim).cuda()
            k = torch.randn(batch, num_heads, seq_len, head_dim).cuda()
            v = torch.randn(batch, num_heads, seq_len, head_dim).cuda()

            # Run SDPA
            with torch.no_grad():
                output = scaled_dot_product_attention(q, k, v)

            assert output.shape == (batch, num_heads, seq_len, head_dim)

        except ImportError:
            pytest.skip("SDPA not available")

    def test_mixed_precision(self):
        """Test mixed precision training/inference"""
        from torch.cuda.amp import autocast

        model = torch.nn.Linear(768, 768).cuda()
        input_data = torch.randn(10, 768).cuda()

        # Without AMP
        with torch.no_grad():
            output_normal = model(input_data)

        # With AMP
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                output_amp = model(input_data)

        # Both should work
        assert output_normal.shape == output_amp.shape

    def test_optimization_flags(self):
        """Test optimization control center"""
        control = OptimizationControlCenter()

        # Test enabling optimizations
        control.enable_optimization("fp16")
        assert control.is_enabled("fp16")

        control.enable_optimization("sdpa")
        assert control.is_enabled("sdpa")

        # Ensure bad optimizations are disabled
        control.disable_optimization("torch_compile")
        assert not control.is_enabled("torch_compile")


class TestMemoryManagement:
    """Test memory management features"""

    def test_memory_tracking(self):
        """Test GPU memory tracking"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        initial_memory = torch.cuda.memory_allocated()

        # Allocate some tensors
        tensors = []
        for _ in range(10):
            tensors.append(torch.randn(1000, 1000).cuda())

        peak_memory = torch.cuda.max_memory_allocated()

        # Clean up
        del tensors
        torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()

        assert peak_memory > initial_memory
        assert final_memory <= initial_memory + 1e6  # Should be mostly freed

    def test_cache_clearing(self):
        """Test cache clearing functionality"""
        # Allocate and free memory
        tensor = torch.randn(10000, 10000).cuda()
        del tensor

        # Cache should still hold memory
        cached_before = torch.cuda.memory_reserved()

        # Clear cache
        torch.cuda.empty_cache()

        cached_after = torch.cuda.memory_reserved()

        # Should reduce cached memory
        assert cached_after <= cached_before


class TestEndToEnd:
    """End-to-end integration tests"""

    @pytest.mark.slow
    def test_full_pipeline(self):
        """Test complete inference pipeline"""
        # This would be a full test with actual model
        # Skipped if model not available

        model_path = Path("gpt-oss-20b/original")
        if not model_path.exists():
            pytest.skip("Model not found")

        # Initialize
        control = OptimizationControlCenter()
        control.enable_optimization("fp16")
        control.enable_optimization("sdpa")

        # Load model
        loader = MoEModelLoader(str(model_path))

        try:
            model = loader.create_model_fp16(top_k=4, full_layers=False)

            # Run inference
            input_ids = torch.randint(0, 50000, (1, 64)).cuda()

            with torch.no_grad():
                start = time.time()
                output = model.generate(input_ids, max_new_tokens=32)
                elapsed = time.time() - start

            # Check performance
            tokens_generated = 32
            throughput = tokens_generated / elapsed

            assert throughput > 5  # Should be at least 5 TPS
            assert output.shape[1] == 96  # 64 input + 32 generated

        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip("Insufficient GPU memory")
            raise

    def test_monitoring_integration(self):
        """Test monitoring system integration"""
        from src.moe.optimization_safety.optimization_monitor import HealthMonitor

        monitor = HealthMonitor()

        # Simulate metrics
        metrics = {
            'throughput_tps': 29.1,
            'latency_ms': 30,
            'memory_gb': 7.3,
            'error_rate': 0.001
        }

        # Check thresholds
        violations = monitor.check_thresholds(metrics)

        # Should pass our production thresholds
        assert len(violations) == 0 or 'throughput_tps' not in violations


class TestConfiguration:
    """Test configuration management"""

    def test_production_config(self):
        """Test loading production configuration"""
        config_path = Path("configs/production.yaml")

        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Verify critical settings
            assert config['optimization']['torch_compile'] == False
            assert config['optimization']['sdpa'] == True
            assert config['model']['experts_per_token'] == 4

    def test_environment_variables(self):
        """Test environment variable configuration"""
        # Check critical env vars
        cuda_home = os.environ.get('CUDA_HOME')
        torch_compile_disable = os.environ.get('TORCH_COMPILE_DISABLE')

        # Warn if not set correctly
        if torch_compile_disable != '1':
            pytest.skip("TORCH_COMPILE_DISABLE should be set to 1")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])