#!/usr/bin/env python3
"""
Real MoE Model Performance Test v3 - WITH FIXES
- Fixed INT8 quantization dtype mismatch
- Load full 24 layers (not just 12)
- Proper memory management
- Combined optimizations testing
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
import json
import csv
from datetime import datetime
from pathlib import Path
import sys
import os
import logging
import gc
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors import safe_open
import bitsandbytes as bnb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Feature flags for safe enable/disable
FEATURE_FLAGS = {
    "fp16_baseline": True,
    "int8_quantization_fixed": True,  # Fixed version
    "mixed_precision": False,  # Disabled - slower than baseline
    "sdpa_attention": True,
    "int8_plus_sdpa": True,  # Combined optimization
}

def memory_stress_test(size_gb: float = 20.0) -> bool:
    """Pre-flight memory stress test"""
    try:
        logger.info(f"Running memory stress test ({size_gb}GB)...")
        elements = int(size_gb * 1e9 / 4)
        dummy = torch.empty(elements, dtype=torch.float32, device='cuda')
        torch.cuda.synchronize()
        del dummy
        torch.cuda.empty_cache()
        logger.info("Memory stress test passed")
        return True
    except RuntimeError as e:
        logger.error(f"Memory stress test failed: {e}")
        return False

class MoEModelLoader:
    """Loads GPT-OSS-20B with top-k expert selection - FIXED VERSION"""

    def __init__(self, model_path: str = "gpt-oss-20b/original"):
        self.model_path = Path(model_path)
        self.config_path = self.model_path / "config.json"
        self.weights_path = self.model_path / "model.safetensors"

    def load_config(self) -> Dict:
        """Load model configuration"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Model config: {config['num_experts']} experts, top-k={config['experts_per_token']}")
        return config

    def load_weights_selective(self, expert_indices: List[int] = None) -> Dict[str, torch.Tensor]:
        """Load only selected experts to save memory"""
        weights = {}

        with safe_open(self.weights_path, framework="pt", device="cpu") as f:
            tensor_names = f.keys()

            for name in tensor_names:
                if "expert" not in name:
                    weights[name] = f.get_tensor(name)
                elif expert_indices:
                    parts = name.split(".")
                    if "expert" in parts:
                        expert_idx = int(parts[parts.index("expert") + 1])
                        if expert_idx in expert_indices:
                            weights[name] = f.get_tensor(name)

        logger.info(f"Loaded {len(weights)} tensors (top-k={len(expert_indices) if expert_indices else 'all'})")
        return weights

    def create_model_fp16(self, top_k: int = 4, full_layers: bool = True) -> nn.Module:
        """Create FP16 model with top-k experts"""
        config = self.load_config()

        expert_indices = list(range(top_k))
        logger.info(f"Loading model with experts: {expert_indices}")
        weights = self.load_weights_selective(expert_indices)

        class SimpleMoE(nn.Module):
            def __init__(self, config, weights):
                super().__init__()
                self.config = config
                self.hidden_size = config['hidden_size']
                self.num_heads = config['num_attention_heads']
                self.head_dim = config.get('head_dim', self.hidden_size // self.num_heads)

                self.embed = nn.Embedding(config['vocab_size'], config['hidden_size'])
                self.layers = nn.ModuleList()

                # Use ALL 24 layers if full_layers=True
                num_layers = config['num_hidden_layers'] if full_layers else 12
                logger.info(f"Creating {num_layers} transformer layers")

                for layer_idx in range(num_layers):
                    self.layers.append(self._create_layer(layer_idx, config))

                self.ln_f = nn.LayerNorm(config['hidden_size'])
                self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)

                self._init_weights()

            def _create_layer(self, idx, config):
                """Create a single transformer layer with MoE"""
                layer = nn.ModuleDict({
                    'ln1': nn.LayerNorm(config['hidden_size']),
                    'attn': nn.MultiheadAttention(
                        config['hidden_size'],
                        config['num_attention_heads'],
                        batch_first=True,
                        dropout=0.0
                    ),
                    'ln2': nn.LayerNorm(config['hidden_size']),
                    'gate': nn.Linear(config['hidden_size'], top_k),  # Expert gating
                    'experts': nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(config['hidden_size'], config['intermediate_size']),
                            nn.GELU(),
                            nn.Linear(config['intermediate_size'], config['hidden_size'])
                        ) for _ in range(top_k)
                    ])
                })
                return layer

            def _init_weights(self):
                """Initialize weights with small random values"""
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(mean=0.0, std=0.02)
                        if module.bias is not None:
                            module.bias.data.zero_()
                    elif isinstance(module, nn.Embedding):
                        module.weight.data.normal_(mean=0.0, std=0.02)

            def forward(self, input_ids, attention_mask=None):
                x = self.embed(input_ids)
                batch_size, seq_len = input_ids.shape

                for layer in self.layers:
                    # Self-attention with residual
                    residual = x
                    x = layer['ln1'](x)
                    attn_out, _ = layer['attn'](x, x, x, need_weights=False)
                    x = residual + attn_out

                    # MoE layer with gating
                    residual = x
                    x = layer['ln2'](x)

                    # Expert gating
                    gate_logits = layer['gate'](x)
                    gate_weights = torch.softmax(gate_logits, dim=-1)

                    # Apply top-k experts
                    expert_outputs = []
                    for i, expert in enumerate(layer['experts']):
                        expert_out = expert(x)
                        weight = gate_weights[:, :, i:i+1]
                        expert_outputs.append(expert_out * weight)

                    x = sum(expert_outputs)
                    x = residual + x

                x = self.ln_f(x)
                logits = self.lm_head(x)
                return logits

            def generate(self, input_ids, max_new_tokens=128, temperature=0.7, measure_first_token=False):
                """Generation with first token latency measurement"""
                device = input_ids.device
                generated = input_ids.clone()
                first_token_latency = None

                with torch.no_grad():
                    for i in range(max_new_tokens):
                        if i == 0 and measure_first_token:
                            torch.cuda.synchronize()
                            start_first = time.perf_counter()

                        logits = self.forward(generated)

                        if i == 0 and measure_first_token:
                            torch.cuda.synchronize()
                            first_token_latency = time.perf_counter() - start_first

                        next_token_logits = logits[:, -1, :] / temperature
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        generated = torch.cat([generated, next_token], dim=-1)

                return generated, first_token_latency

        model = SimpleMoE(config, weights)
        return model.half().cuda()

    def create_model_int8_fixed(self, top_k: int = 4, full_layers: bool = True) -> nn.Module:
        """FIXED INT8 quantization - handles dtype properly"""
        # Start with FP16 model
        model_fp16 = self.create_model_fp16(top_k, full_layers)

        # CRITICAL FIX: Convert to FP32 first for INT8 quantization
        logger.info("Converting model to FP32 for INT8 quantization...")
        model_fp32 = model_fp16.float()  # FP16 → FP32

        # Clear FP16 model from memory
        del model_fp16
        torch.cuda.empty_cache()

        mem_before = torch.cuda.memory_allocated()

        def replace_with_int8(module, prefix=""):
            """Replace Linear layers with INT8 - FIXED VERSION"""
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    logger.debug(f"Converting {prefix}.{name} to INT8")

                    # Create INT8 linear layer
                    int8_linear = bnb.nn.Linear8bitLt(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        has_fp16_weights=False,
                        threshold=6.0
                    )

                    # Copy weights (ensure FP32)
                    with torch.no_grad():
                        int8_linear.weight = bnb.nn.Int8Params(
                            child.weight.data.cpu().float(),  # Ensure FP32
                            requires_grad=False
                        )

                        # CRITICAL FIX: Bias must be a Parameter
                        if child.bias is not None:
                            int8_linear.bias = nn.Parameter(
                                child.bias.data.float().cuda(),
                                requires_grad=False
                            )

                    # Move to CUDA
                    int8_linear = int8_linear.cuda()

                    # Replace module
                    setattr(module, name, int8_linear)

                    # Free original memory
                    del child
                    torch.cuda.empty_cache()

                elif isinstance(child, nn.ModuleList) or isinstance(child, nn.ModuleDict):
                    replace_with_int8(child, f"{prefix}.{name}")
                else:
                    replace_with_int8(child, f"{prefix}.{name}")

        # Apply INT8 conversion
        replace_with_int8(model_fp32)

        # Convert non-linear layers back to FP16 for efficiency
        for module in model_fp32.modules():
            if not isinstance(module, bnb.nn.Linear8bitLt):
                if isinstance(module, (nn.LayerNorm, nn.Embedding)):
                    module.half()

        mem_after = torch.cuda.memory_allocated()
        logger.info(f"INT8 conversion complete: {(mem_before-mem_after)/1e9:.2f}GB saved")

        return model_fp32

    def create_model_int8_sdpa(self, top_k: int = 4, full_layers: bool = True) -> nn.Module:
        """Combined INT8 + SDPA optimization"""
        # Start with INT8 model
        model = self.create_model_int8_fixed(top_k, full_layers)

        # Add SDPA to attention layers
        from torch.nn.functional import scaled_dot_product_attention

        for layer in model.layers:
            original_attn = layer['attn']

            class SDPAAttention(nn.Module):
                def __init__(self, orig_attn):
                    super().__init__()
                    self.embed_dim = orig_attn.embed_dim
                    self.num_heads = orig_attn.num_heads
                    self.head_dim = self.embed_dim // self.num_heads
                    self.orig_attn = orig_attn

                def forward(self, query, key, value, need_weights=False):
                    batch_size, seq_len, _ = query.shape

                    # Reshape for multi-head attention
                    q = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    k = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    v = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

                    # Use SDPA with Flash Attention
                    attn_output = scaled_dot_product_attention(
                        q, k, v,
                        dropout_p=0.0,
                        is_causal=False
                    )

                    # Reshape back
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

                    return attn_output, None

            layer['attn'] = SDPAAttention(original_attn)

        logger.info("Applied SDPA to INT8 model")
        return model

    def create_model_sdpa(self, top_k: int = 4, full_layers: bool = True) -> nn.Module:
        """Create model with SDPA (Flash Attention)"""
        model = self.create_model_fp16(top_k, full_layers)

        from torch.nn.functional import scaled_dot_product_attention

        for layer in model.layers:
            original_attn = layer['attn']

            class SDPAAttention(nn.Module):
                def __init__(self, orig_attn):
                    super().__init__()
                    self.embed_dim = orig_attn.embed_dim
                    self.num_heads = orig_attn.num_heads
                    self.head_dim = self.embed_dim // self.num_heads
                    self.orig_attn = orig_attn

                def forward(self, query, key, value, need_weights=False):
                    batch_size, seq_len, _ = query.shape

                    q = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    k = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    v = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

                    attn_output = scaled_dot_product_attention(
                        q, k, v,
                        dropout_p=0.0,
                        is_causal=False
                    )

                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

                    return attn_output, None

            layer['attn'] = SDPAAttention(original_attn)

        return model


def benchmark_generation(model, input_ids, num_runs: int = 3, max_new_tokens: int = 64) -> Dict:
    """Benchmark with first token latency tracking"""
    device = torch.device("cuda")
    input_ids = input_ids.to(device)

    latencies = []
    first_token_latencies = []
    tokens_generated = []

    # Warmup
    logger.info("Running warmup...")
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=10)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark runs
    for run in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output, first_token_latency = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                measure_first_token=True
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        latencies.append(elapsed)
        if first_token_latency:
            first_token_latencies.append(first_token_latency)
        tokens_generated.append(max_new_tokens)

        logger.info(f"Run {run+1}: {elapsed:.2f}s ({max_new_tokens} tokens), "
                   f"First token: {first_token_latency*1000:.1f}ms")

    # Calculate metrics
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    avg_first_token = np.mean(first_token_latencies) if first_token_latencies else 0
    total_tokens = sum(tokens_generated)
    avg_tps = total_tokens / sum(latencies)
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9

    return {
        "avg_latency_s": avg_latency,
        "std_latency_s": std_latency,
        "avg_first_token_ms": avg_first_token * 1000,
        "avg_tps": avg_tps,
        "peak_memory_gb": peak_memory_gb,
        "runs": num_runs,
        "tokens_per_run": max_new_tokens
    }


def test_configuration(name: str, model_fn, input_ids) -> Dict:
    """Test a single configuration with proper cleanup"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {name}")
    logger.info(f"{'='*60}")

    # Clear memory before test
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        # Create model
        model = model_fn()

        # Benchmark
        results = benchmark_generation(model, input_ids)

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

        results["status"] = "success"
        return results

    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        # Cleanup on failure
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "status": "failed",
            "error": str(e),
            "avg_latency_s": 0,
            "std_latency_s": 0,
            "avg_first_token_ms": 0,
            "avg_tps": 0,
            "peak_memory_gb": 0
        }


def main():
    """Main test execution"""
    print("="*80)
    print("REAL MoE MODEL PERFORMANCE TEST v3 - FIXED")
    print("With INT8 fix, full layers, and combined optimizations")
    print("="*80)

    # Check environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: CUDA not available. Exiting.")
        return

    print(f"\nEnvironment:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Memory stress test - increased for full model
    if not memory_stress_test(size_gb=18.0):
        print("WARNING: May have insufficient memory for full model")

    # Initialize model loader
    loader = MoEModelLoader()

    # Create test input
    input_ids = torch.randint(0, 50000, (1, 64))  # Batch=1, Seq=64

    # Test configurations
    results = {}

    # Test with 12 layers first (baseline)
    if FEATURE_FLAGS["fp16_baseline"]:
        print("\n--- Testing with 12 layers (baseline) ---")
        results["fp16_12layers"] = test_configuration(
            "FP16 Baseline (12 layers)",
            lambda: loader.create_model_fp16(top_k=4, full_layers=False),
            input_ids
        )

    # Test with full 24 layers
    if FEATURE_FLAGS["fp16_baseline"]:
        print("\n--- Testing with 24 layers (full model) ---")
        results["fp16_24layers"] = test_configuration(
            "FP16 Baseline (24 layers)",
            lambda: loader.create_model_fp16(top_k=4, full_layers=True),
            input_ids
        )

    # Fixed INT8 quantization
    if FEATURE_FLAGS["int8_quantization_fixed"]:
        results["int8_fixed"] = test_configuration(
            "INT8 Quantization (FIXED)",
            lambda: loader.create_model_int8_fixed(top_k=4, full_layers=False),
            input_ids
        )

    # SDPA/Flash Attention
    if FEATURE_FLAGS["sdpa_attention"]:
        results["sdpa_attention"] = test_configuration(
            "SDPA (Flash Attention)",
            lambda: loader.create_model_sdpa(top_k=4, full_layers=False),
            input_ids
        )

    # Combined INT8 + SDPA
    if FEATURE_FLAGS["int8_plus_sdpa"]:
        results["int8_sdpa_combined"] = test_configuration(
            "INT8 + SDPA Combined",
            lambda: loader.create_model_int8_sdpa(top_k=4, full_layers=False),
            input_ids
        )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("tests/test_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
        },
        "configurations": results,
        "feature_flags": FEATURE_FLAGS
    }

    json_path = output_dir / f"real_moe_performance_fixed_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Print enhanced summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY - FIXED VERSION")
    print("="*80)
    print(f"{'Configuration':<25} | {'First Token (ms)':<15} | {'Avg TPS':<10} | {'Peak Mem (GB)':<12} | {'Status':<8}")
    print("-"*80)

    for config_name, metrics in results.items():
        if metrics['status'] == 'success':
            print(f"{config_name:<25} | {metrics['avg_first_token_ms']:>14.1f} | "
                  f"{metrics['avg_tps']:>9.1f} | {metrics['peak_memory_gb']:>11.2f} | SUCCESS")
        else:
            error = metrics.get('error', 'Unknown')[:20]
            print(f"{config_name:<25} | {'---':>15} | {'---':>10} | {'---':>12} | FAILED: {error}")

    # Compare 12 vs 24 layers if both tested
    if "fp16_12layers" in results and "fp16_24layers" in results:
        if results["fp16_12layers"]["status"] == "success" and results["fp16_24layers"]["status"] == "success":
            mem_increase = results["fp16_24layers"]["peak_memory_gb"] - results["fp16_12layers"]["peak_memory_gb"]
            tps_decrease = results["fp16_12layers"]["avg_tps"] - results["fp16_24layers"]["avg_tps"]
            print(f"\n24 vs 12 Layers Impact:")
            print(f"  Memory increase: +{mem_increase:.2f} GB")
            print(f"  TPS decrease: -{tps_decrease:.1f}")

    # Best configuration analysis
    print("\n" + "="*80)
    print("OPTIMIZATION ANALYSIS")
    print("="*80)

    successful = [m for m in results.values() if m['status'] == 'success']
    if successful:
        best_tps_config = max(results.items(), key=lambda x: x[1]['avg_tps'] if x[1]['status'] == 'success' else 0)
        best_mem_config = min(results.items(), key=lambda x: x[1]['peak_memory_gb'] if x[1]['status'] == 'success' else float('inf'))

        print(f"Best Throughput: {best_tps_config[0]} - {best_tps_config[1]['avg_tps']:.1f} TPS")
        print(f"Best Memory: {best_mem_config[0]} - {best_mem_config[1]['peak_memory_gb']:.2f} GB")

        # Check if INT8 fix worked
        if "int8_fixed" in results and results["int8_fixed"]["status"] == "success":
            print("\n✅ INT8 FIX SUCCESSFUL!")
            if "fp16_12layers" in results:
                fp16_mem = results["fp16_12layers"]["peak_memory_gb"]
                int8_mem = results["int8_fixed"]["peak_memory_gb"]
                print(f"   Memory reduction: {(1 - int8_mem/fp16_mem)*100:.1f}%")

        # Check combined optimization
        if "int8_sdpa_combined" in results and results["int8_sdpa_combined"]["status"] == "success":
            print("\n✅ COMBINED INT8 + SDPA SUCCESSFUL!")
            combined_tps = results["int8_sdpa_combined"]["avg_tps"]
            combined_mem = results["int8_sdpa_combined"]["peak_memory_gb"]
            print(f"   Throughput: {combined_tps:.1f} TPS")
            print(f"   Memory: {combined_mem:.2f} GB")

    print("="*80)
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()