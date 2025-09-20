#!/usr/bin/env python3
"""
MoE Configuration System with Feature Flags
Central configuration for all optimizations with safe defaults
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CUDAKernelConfig:
    """CUDA kernel fusion configuration"""
    enabled: bool = False  # Feature flag - default OFF for safety
    fallback_on_error: bool = True
    numerical_tolerance: float = 1e-6
    gradient_tolerance: float = 1e-5
    use_triton: bool = True  # Use Triton vs native CUDA


@dataclass
class AsyncIOConfig:
    """Async I/O configuration"""
    enabled: bool = False  # Feature flag - default OFF
    prefetch_window: int = 3
    timeout_ms: int = 100
    max_concurrent_loads: int = 8
    fallback_to_sync: bool = True


@dataclass
class CacheConfig:
    """Tiered caching configuration"""
    mode: Literal["single", "tiered"] = "single"  # Default to simple cache
    gpu_capacity_gb: float = 2.0
    ram_capacity_gb: float = 16.0
    disk_capacity_gb: float = 100.0
    eviction_policy: Literal["lru", "arc", "lfu"] = "lru"
    enable_prefetch: bool = False


@dataclass
class MultiGPUConfig:
    """Multi-GPU parallelization configuration"""
    enabled: bool = False  # Feature flag - default OFF
    world_size: Optional[int] = None  # Auto-detect if None
    nccl_timeout_seconds: int = 30
    fallback_single_gpu: bool = True
    expert_distribution: Literal["balanced", "dynamic"] = "balanced"
    pipeline_parallel: bool = False


@dataclass
class OptimizationConfig:
    """Quick win optimizations"""
    dynamic_batching: bool = False
    memory_pools: bool = False
    router_batch_ops: bool = False
    gradient_checkpointing: bool = False
    compile_mode: Optional[str] = None  # torch.compile mode


@dataclass
class MoEConfig:
    """Complete MoE configuration"""
    # Model parameters
    model_path: str = "./gpt-oss-20b"
    num_layers: int = 24
    num_experts: int = 32
    experts_per_token: int = 4
    hidden_dim: int = 2880

    # Device configuration
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 42

    # Optimization configs
    cuda_kernels: CUDAKernelConfig = field(default_factory=CUDAKernelConfig)
    async_io: AsyncIOConfig = field(default_factory=AsyncIOConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    multi_gpu: MultiGPUConfig = field(default_factory=MultiGPUConfig)
    optimizations: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Monitoring
    enable_profiling: bool = False
    enable_metrics: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str) -> "MoEConfig":
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse nested configs
        config = cls()

        if 'cuda_kernels' in data:
            config.cuda_kernels = CUDAKernelConfig(**data['cuda_kernels'])
        if 'async_io' in data:
            config.async_io = AsyncIOConfig(**data['async_io'])
        if 'cache' in data:
            config.cache = CacheConfig(**data['cache'])
        if 'multi_gpu' in data:
            config.multi_gpu = MultiGPUConfig(**data['multi_gpu'])
        if 'optimizations' in data:
            config.optimizations = OptimizationConfig(**data['optimizations'])

        # Set top-level attributes
        for key in ['model_path', 'num_layers', 'num_experts', 'device', 'dtype']:
            if key in data:
                setattr(config, key, data[key])

        return config

    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        data = {
            'model_path': self.model_path,
            'num_layers': self.num_layers,
            'num_experts': self.num_experts,
            'experts_per_token': self.experts_per_token,
            'hidden_dim': self.hidden_dim,
            'device': self.device,
            'dtype': self.dtype,
            'seed': self.seed,
            'cuda_kernels': {
                'enabled': self.cuda_kernels.enabled,
                'fallback_on_error': self.cuda_kernels.fallback_on_error,
                'numerical_tolerance': self.cuda_kernels.numerical_tolerance,
                'gradient_tolerance': self.cuda_kernels.gradient_tolerance,
                'use_triton': self.cuda_kernels.use_triton,
            },
            'async_io': {
                'enabled': self.async_io.enabled,
                'prefetch_window': self.async_io.prefetch_window,
                'timeout_ms': self.async_io.timeout_ms,
                'max_concurrent_loads': self.async_io.max_concurrent_loads,
                'fallback_to_sync': self.async_io.fallback_to_sync,
            },
            'cache': {
                'mode': self.cache.mode,
                'gpu_capacity_gb': self.cache.gpu_capacity_gb,
                'ram_capacity_gb': self.cache.ram_capacity_gb,
                'disk_capacity_gb': self.cache.disk_capacity_gb,
                'eviction_policy': self.cache.eviction_policy,
                'enable_prefetch': self.cache.enable_prefetch,
            },
            'multi_gpu': {
                'enabled': self.multi_gpu.enabled,
                'world_size': self.multi_gpu.world_size,
                'nccl_timeout_seconds': self.multi_gpu.nccl_timeout_seconds,
                'fallback_single_gpu': self.multi_gpu.fallback_single_gpu,
                'expert_distribution': self.multi_gpu.expert_distribution,
                'pipeline_parallel': self.multi_gpu.pipeline_parallel,
            },
            'optimizations': {
                'dynamic_batching': self.optimizations.dynamic_batching,
                'memory_pools': self.optimizations.memory_pools,
                'router_batch_ops': self.optimizations.router_batch_ops,
                'gradient_checkpointing': self.optimizations.gradient_checkpointing,
                'compile_mode': self.optimizations.compile_mode,
            },
            'enable_profiling': self.enable_profiling,
            'enable_metrics': self.enable_metrics,
            'log_level': self.log_level,
        }

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def validate(self) -> bool:
        """Validate configuration consistency"""
        valid = True

        # Check GPU memory constraints
        if self.cache.gpu_capacity_gb > 24:  # RTX 3090 limit
            logger.warning(f"GPU cache {self.cache.gpu_capacity_gb}GB exceeds typical limits")
            valid = False

        # Check multi-GPU consistency
        if self.multi_gpu.enabled and self.device == "cpu":
            logger.error("Multi-GPU enabled but device is CPU")
            valid = False

        # Check async I/O dependencies
        if self.async_io.enabled and self.cache.mode == "single":
            logger.info("Async I/O works best with tiered caching")

        # Warn about experimental features
        if self.cuda_kernels.enabled:
            logger.warning("CUDA kernel fusion is experimental")

        return valid

    def get_active_optimizations(self) -> list:
        """Return list of enabled optimizations"""
        active = []

        if self.cuda_kernels.enabled:
            active.append("CUDA Kernel Fusion")
        if self.async_io.enabled:
            active.append("Async I/O")
        if self.cache.mode == "tiered":
            active.append("Tiered Caching")
        if self.multi_gpu.enabled:
            active.append("Multi-GPU")
        if self.optimizations.dynamic_batching:
            active.append("Dynamic Batching")
        if self.optimizations.memory_pools:
            active.append("Memory Pools")
        if self.optimizations.router_batch_ops:
            active.append("Router Batch Ops")

        return active


# Default configuration (all optimizations OFF for safety)
DEFAULT_CONFIG = MoEConfig()


def load_config(path: Optional[str] = None) -> MoEConfig:
    """Load configuration from file or use defaults"""
    if path and Path(path).exists():
        logger.info(f"Loading config from {path}")
        config = MoEConfig.from_yaml(path)
    else:
        logger.info("Using default configuration (all optimizations OFF)")
        config = DEFAULT_CONFIG

    # Validate
    if not config.validate():
        logger.warning("Configuration validation failed, using defaults")
        config = DEFAULT_CONFIG

    # Log active optimizations
    active = config.get_active_optimizations()
    if active:
        logger.info(f"Active optimizations: {', '.join(active)}")
    else:
        logger.info("No optimizations enabled (baseline mode)")

    return config


if __name__ == "__main__":
    # Example: Create and save default config
    config = MoEConfig()
    config.to_yaml("moe_config_default.yaml")
    print("Default configuration saved to moe_config_default.yaml")

    # Example: Enable some optimizations
    config.async_io.enabled = True
    config.cache.mode = "tiered"
    config.to_yaml("moe_config_optimized.yaml")
    print("Optimized configuration saved to moe_config_optimized.yaml")

    # Show active optimizations
    print(f"Active optimizations: {config.get_active_optimizations()}")