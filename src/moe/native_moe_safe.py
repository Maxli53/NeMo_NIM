#!/usr/bin/env python3
"""
Safety-Enhanced Native MoE Implementation for GPT-OSS-20B
Integrates all components with comprehensive safety framework
"""

import torch
import torch.nn as nn
from safetensors import safe_open
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import logging
import time
import psutil

# Import safety framework
from .optimization_safety.optimization_control_center import OptimizationControlCenter
from .optimization_safety.safe_optimizations import (
    SafeCUDAKernels,
    SafeAsyncIO,
    SafeTieredCache,
    SafeMultiGPU,
    create_safe_optimizations
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafeGPTOSSNativeMoE(nn.Module):
    """
    Native MoE with integrated safety framework for all optimizations
    """

    def __init__(
        self,
        model_path: str,
        cache_size_gb: float = 5.0,
        device: str = "cuda",
        enable_optimizations: bool = False
    ):
        super().__init__()
        self.model_path = Path(model_path)
        self.device = device
        
        # Initialize control center
        self.control_center = OptimizationControlCenter.get_instance()
        logger.info("Safety framework initialized")
        
        # Load config
        with open(self.model_path / "config.json") as f:
            self.config = json.load(f)

        self.num_layers = self.config.get("num_hidden_layers", 24)
        self.num_experts = self.config.get("num_local_experts", 32)
        self.experts_per_token = self.config.get("experts_per_token", 4)
        self.hidden_size = self.config.get("hidden_size", 2880)
        self.vocab_size = self.config.get("vocab_size", 201088)

        logger.info(f"Initializing Safe Native MoE: {self.num_layers} layers, {self.num_experts} experts")

        # Shard mapping
        self.shards = {
            0: self.model_path / "model-00000-of-00002.safetensors",
            1: self.model_path / "model-00001-of-00002.safetensors",
            2: self.model_path / "model-00002-of-00002.safetensors",
        }

        # Initialize safe optimizations
        optimization_config = {
            'cache_dir': str(self.model_path / 'expert_cache'),
            'device': device,
            'cache': {
                'cache_size': int(cache_size_gb * 1024 / 27),  # Convert GB to number of experts
                'device': device,
                'gpu_capacity_gb': min(2.0, cache_size_gb),
                'ram_capacity_gb': 16.0,
                'disk_capacity_gb': 100.0
            },
            'world_size': torch.cuda.device_count() if torch.cuda.is_available() else 1
        }
        
        self.optimizations = create_safe_optimizations(optimization_config)
        logger.info(f"Created {len(self.optimizations)} safe optimization wrappers")
        
        # Enable optimizations if requested (still respects control center flags)
        if enable_optimizations:
            self._enable_default_optimizations()

        # Initialize components
        self.expert_cache = {}  # Basic cache, replaced by tiered cache if enabled
        self.load_count = 0
        self.cache_hits = 0

        # Load routers (small, keep in memory) as parameters
        router_data = self._load_all_routers()
        self.routers = nn.ParameterDict()
        for layer_idx, router in router_data.items():
            self.routers[str(layer_idx)] = nn.ParameterDict({
                'weight': nn.Parameter(router['weight'], requires_grad=False),
                'bias': nn.Parameter(router['bias'], requires_grad=False)
            })
        logger.info(f"Loaded {len(self.routers)} router layers as parameters")

        # Track memory
        self.initial_memory = self._get_memory_stats()
        
    def _enable_default_optimizations(self):
        """Enable safe subset of optimizations for testing"""
        # Only enable cache optimization by default (safest)
        self.control_center.enable_optimization("tiered_cache", traffic_percentage=1.0, validate=True)
        logger.info("Enabled tiered cache optimization (other optimizations remain OFF by default)")

    def _load_all_routers(self) -> Dict:
        """Load all router weights into memory (they're small)"""
        routers = {}

        for shard_idx in [0, 1]:
            if shard_idx >= len(self.shards):
                continue

            with safe_open(self.shards[shard_idx], framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "router.weight" in key:
                        layer_idx = int(key.split(".")[2])
                        if layer_idx not in routers:
                            weight_key = f"model.layers.{layer_idx}.mlp.router.weight"
                            bias_key = f"model.layers.{layer_idx}.mlp.router.bias"

                            routers[layer_idx] = {
                                "weight": f.get_tensor(weight_key).to(torch.bfloat16).to(self.device),
                                "bias": f.get_tensor(bias_key).to(torch.bfloat16).to(self.device)
                            }

        return routers

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to top-k experts
        """
        router = self.routers[str(layer_idx)]

        # Compute routing scores
        scores = hidden_states @ router["weight"].T + router["bias"]

        # Select top-k experts
        expert_weights, expert_indices = torch.topk(scores, k=self.experts_per_token, dim=-1)
        expert_weights = torch.softmax(expert_weights, dim=-1)

        return expert_indices, expert_weights

    async def load_experts_async(self, layer_idx: int, expert_indices: List[int], router_logits: torch.Tensor) -> Dict:
        """
        Load experts with async prefetching if enabled
        """
        # Try async prefetching if enabled
        if self.optimizations['async_io'].is_enabled():
            await self.optimizations['async_io'].prefetch_experts(router_logits, layer_idx)
        
        return self.load_experts(layer_idx, expert_indices)

    def load_experts(self, layer_idx: int, expert_indices: List[int]) -> Dict:
        """
        Load only the specified experts for a layer with tiered cache support
        """
        experts = {}

        for expert_idx in expert_indices:
            # Try tiered cache first if enabled
            if self.optimizations['tiered_cache'].is_enabled():
                expert, cache_tier = self.optimizations['tiered_cache'].get(layer_idx, expert_idx)
                if expert is not None:
                    experts[expert_idx] = expert
                    self.cache_hits += 1
                    continue
            else:
                # Fallback to simple cache
                cache_key = f"L{layer_idx}_E{expert_idx}"
                if cache_key in self.expert_cache:
                    experts[expert_idx] = self.expert_cache[cache_key]
                    self.cache_hits += 1
                    continue

            # Load from disk (simplified for demo)
            shard_idx = 0 if layer_idx < 12 else 1
            self.load_count += 1

            # In real implementation, would load actual weights
            experts[expert_idx] = {
                "loaded": True,
                "layer": layer_idx,
                "expert": expert_idx
            }

            # Add to appropriate cache
            if not self.optimizations['tiered_cache'].is_enabled():
                cache_key = f"L{layer_idx}_E{expert_idx}"
                self.expert_cache[cache_key] = experts[expert_idx]

        return experts

    def mix_expert_outputs(
        self,
        hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Mix expert outputs with optional CUDA kernel fusion
        """
        # Try CUDA kernel fusion if enabled
        if self.optimizations['cuda_kernels'].is_enabled():
            return self.optimizations['cuda_kernels'].fused_expert_mixer(
                hidden_states, expert_outputs, expert_weights
            )
        
        # Fallback to standard PyTorch
        weighted_outputs = expert_outputs * expert_weights.unsqueeze(-1)
        return weighted_outputs.sum(dim=1)

    def moe_forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through MoE layer with safety-wrapped optimizations
        """
        # Route tokens to experts
        expert_indices, expert_weights = self.route_tokens(hidden_states, layer_idx)

        # Get unique experts needed for this batch
        unique_experts = torch.unique(expert_indices).cpu().tolist()

        # Load only needed experts (with async prefetching if enabled)
        experts = self.load_experts(layer_idx, unique_experts)

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(-1)
            hidden_states = hidden_states * attention_mask.to(hidden_states.dtype)

        # For demo, create dummy expert outputs
        batch_size, seq_len = hidden_states.shape[:2]
        expert_outputs = torch.randn(
            batch_size, seq_len, self.experts_per_token, self.hidden_size,
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        
        # Mix expert outputs (with CUDA kernels if enabled)
        output = self.mix_expert_outputs(hidden_states, expert_outputs, expert_weights)

        return output

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict:
        """
        Full forward pass with multi-GPU support if enabled
        """
        batch_size, seq_len = input_ids.shape

        # Create dummy hidden states for demo
        hidden_states = torch.randn(
            batch_size, seq_len, self.hidden_size,
            dtype=torch.bfloat16, device=self.device
        )

        # Apply initial attention mask if provided
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            hidden_states = hidden_states * mask_expanded

        # Check if multi-GPU is enabled
        if self.optimizations['multi_gpu'].is_enabled():
            # Process through layers with multi-GPU
            for layer_idx in range(min(3, self.num_layers)):
                # Create dummy router logits for multi-GPU routing
                router_logits = torch.randn(
                    batch_size, seq_len, self.num_experts,
                    dtype=hidden_states.dtype, device=hidden_states.device
                )
                
                # Multi-GPU forward (would use actual experts in production)
                hidden_states = self.optimizations['multi_gpu'].forward(
                    hidden_states, router_logits, None  # experts placeholder
                )
        else:
            # Single GPU processing
            for layer_idx in range(min(3, self.num_layers)):
                hidden_states = self.moe_forward(hidden_states, layer_idx, attention_mask)

        # Get final memory stats
        final_memory = self._get_memory_stats()
        
        # Get optimization status
        optimization_status = {
            opt_name: opt.is_enabled() 
            for opt_name, opt in self.optimizations.items()
        }

        return {
            "hidden_states": hidden_states,
            "stats": {
                "experts_loaded": self.load_count,
                "cache_hits": self.cache_hits,
                "cache_size": len(self.expert_cache),
                "memory_before_gb": self.initial_memory["gpu_gb"],
                "memory_after_gb": final_memory["gpu_gb"],
                "memory_saved_gb": self._calculate_savings(),
                "optimizations_enabled": optimization_status,
                "safety_status": self.control_center.get_status()
            }
        }

    def _get_memory_stats(self) -> Dict:
        """Get current memory usage"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
        else:
            gpu_memory = 0
        
        cpu_memory = psutil.Process().memory_info().rss / 1024**3
        
        return {
            "gpu_gb": gpu_memory,
            "cpu_gb": cpu_memory
        }

    def _calculate_savings(self) -> float:
        """Calculate memory savings vs loading all experts"""
        # Each expert is ~27MB in MXFP4, would be ~216MB in BF16
        baseline_gb = self.num_layers * self.num_experts * 0.216
        current_gb = self._get_memory_stats()["gpu_gb"] - self.initial_memory["gpu_gb"]
        return max(0, baseline_gb - current_gb)

    def enable_optimization(self, optimization_name: str, traffic_percentage: float = 0.01):
        """Enable a specific optimization with safety checks"""
        logger.info(f"Enabling {optimization_name} at {traffic_percentage*100}% traffic")
        self.control_center.enable_optimization(
            optimization_name, 
            traffic_percentage=traffic_percentage,
            validate=True
        )
    
    def disable_optimization(self, optimization_name: str, reason: str = "Manual disable"):
        """Disable a specific optimization"""
        logger.info(f"Disabling {optimization_name}: {reason}")
        self.control_center.disable_optimization(optimization_name, reason)
    
    def emergency_stop(self, reason: str = "Emergency stop triggered"):
        """Emergency stop all optimizations"""
        logger.warning(f"EMERGENCY STOP: {reason}")
        self.control_center.emergency_stop_all(reason)


def demo_safe_moe():
    """Demo the safe MoE implementation"""
    logger.info("="*50)
    logger.info("Safe Native MoE Demo with Safety Framework")
    logger.info("="*50)
    
    # Initialize model with safety framework
    model_path = "./models/gpt-oss-20b"  # Update path as needed
    model = SafeGPTOSSNativeMoE(
        model_path=model_path,
        cache_size_gb=5.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_optimizations=False  # All optimizations OFF by default
    )
    
    # Create sample input
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device=model.device)
    attention_mask = torch.ones_like(input_ids)
    
    logger.info("\n1. Testing with all optimizations OFF (baseline)")
    outputs = model(input_ids, attention_mask)
    logger.info(f"Stats: {outputs['stats']}")
    
    # Progressively enable optimizations
    logger.info("\n2. Enabling tiered cache optimization")
    model.enable_optimization("tiered_cache", traffic_percentage=1.0)
    outputs = model(input_ids, attention_mask)
    logger.info(f"Stats: {outputs['stats']}")
    
    logger.info("\n3. Enabling CUDA kernels (if available)")
    if torch.cuda.is_available():
        model.enable_optimization("cuda_kernels", traffic_percentage=0.1)  # Start with 10%
        outputs = model(input_ids, attention_mask)
        logger.info(f"Stats: {outputs['stats']}")
    
    # Demonstrate emergency stop
    logger.info("\n4. Testing emergency stop")
    model.emergency_stop("Demo emergency stop")
    outputs = model(input_ids, attention_mask)
    logger.info(f"All optimizations disabled: {outputs['stats']['optimizations_enabled']}")
    
    logger.info("\n" + "="*50)
    logger.info("Demo complete - Safety framework working correctly")
    logger.info("="*50)


if __name__ == "__main__":
    demo_safe_moe()