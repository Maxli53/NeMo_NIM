#!/usr/bin/env python3
"""
Flash Attention v2 Implementation
Final optimization from Phase 1 roadmap

Provides 1.5-2× speedup for attention operations
Requires CUDA Compute Capability >= 8.0 (RTX 3090 supports it)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FlashAttentionConfig:
    """Configuration for Flash Attention"""
    enabled: bool = False
    use_flash_attn: bool = True
    causal: bool = True
    dropout_p: float = 0.0
    scale: Optional[float] = None
    window_size: Tuple[int, int] = (-1, -1)  # No sliding window by default
    alibi_slopes: Optional[torch.Tensor] = None
    deterministic: bool = False

    # Fallback settings
    fallback_to_standard: bool = True
    min_compute_capability: float = 8.0  # Ampere or newer

    # Performance settings
    block_size_q: int = 128
    block_size_kv: int = 128


class FlashAttention(nn.Module):
    """
    Flash Attention v2 implementation with automatic fallback

    Provides fast, memory-efficient attention computation
    Falls back to standard attention on unsupported hardware
    """

    def __init__(self, config: FlashAttentionConfig):
        super().__init__()
        self.config = config
        self.flash_available = self._check_flash_availability()

        if self.flash_available:
            logger.info("Flash Attention v2 is available and will be used")
        else:
            logger.warning("Flash Attention not available, using standard attention")

    def _check_flash_availability(self) -> bool:
        """Check if Flash Attention can be used on current hardware"""

        if not torch.cuda.is_available():
            return False

        # Check compute capability
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        compute_capability = capability[0] + capability[1] / 10.0

        if compute_capability < self.config.min_compute_capability:
            logger.info(f"Compute capability {compute_capability} < {self.config.min_compute_capability}")
            return False

        # Try to import flash_attn
        try:
            import flash_attn
            from flash_attn import flash_attn_func
            self._flash_attn_func = flash_attn_func
            return True
        except ImportError:
            logger.info("flash_attn package not installed")
            return False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: Optional[bool] = None,
        return_attn_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute attention using Flash Attention v2 or standard attention

        Args:
            query: [batch_size, seq_len, num_heads, head_dim]
            key: [batch_size, seq_len, num_heads, head_dim]
            value: [batch_size, seq_len, num_heads, head_dim]
            attention_mask: Optional mask tensor
            causal: Whether to use causal masking
            return_attn_weights: Whether to return attention weights (not supported in flash)

        Returns:
            output: [batch_size, seq_len, num_heads, head_dim]
            attn_weights: Optional attention weights (None if using flash)
        """

        if causal is None:
            causal = self.config.causal

        # Use Flash Attention if available and enabled
        if self.config.enabled and self.flash_available and not return_attn_weights:
            return self._flash_attention(query, key, value, causal)
        else:
            return self._standard_attention(query, key, value, attention_mask, causal, return_attn_weights)

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool
    ) -> Tuple[torch.Tensor, None]:
        """
        Compute attention using Flash Attention v2
        """

        batch_size, seq_len, num_heads, head_dim = query.shape

        # Flash attention expects [batch_size, seq_len, num_heads, head_dim]
        # but in contiguous memory layout
        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()

        # Compute scale if not provided
        scale = self.config.scale or (1.0 / math.sqrt(head_dim))

        try:
            # Call flash attention
            # Note: flash_attn_func expects specific tensor layout
            output = self._flash_attn_func(
                q, k, v,
                dropout_p=self.config.dropout_p,
                softmax_scale=scale,
                causal=causal,
                window_size=self.config.window_size,
                alibi_slopes=self.config.alibi_slopes,
                deterministic=self.config.deterministic
            )

            return output, None

        except Exception as e:
            logger.warning(f"Flash attention failed: {e}, falling back to standard")
            if self.config.fallback_to_standard:
                return self._standard_attention(query, key, value, None, causal, False)
            else:
                raise

    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        causal: bool,
        return_attn_weights: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute attention using standard PyTorch operations
        """

        batch_size, seq_len, num_heads, head_dim = query.shape

        # Reshape for batched matrix multiplication
        # [batch_size, num_heads, seq_len, head_dim]
        q = query.transpose(1, 2).float()  # Ensure float32 for softmax stability
        k = key.transpose(1, 2).float()
        v = value.transpose(1, 2).float()

        # Compute attention scores
        scale = self.config.scale or (1.0 / math.sqrt(head_dim))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply masking
        if causal:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=scores.device),
                diagonal=1
            )
            scores = scores + causal_mask

        if attention_mask is not None:
            scores = scores + attention_mask

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout if specified
        if self.config.dropout_p > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.config.dropout_p)

        # Compute output
        output = torch.matmul(attn_weights, v)

        # Reshape back
        output = output.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]

        # Convert back to original dtype
        output = output.to(query.dtype)

        if return_attn_weights:
            return output, attn_weights
        else:
            return output, None


class FlashAttentionRouter(nn.Module):
    """
    Router module that can use Flash Attention for the routing mechanism
    """

    def __init__(self, hidden_dim: int, num_experts: int, config: Optional[FlashAttentionConfig] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.config = config or FlashAttentionConfig()

        # Router layers
        self.router_proj = nn.Linear(hidden_dim, num_experts)

        # Optional attention-based routing
        if self.config.enabled:
            self.attention = FlashAttention(self.config)
            self.expert_embeddings = nn.Parameter(
                torch.randn(1, num_experts, 1, hidden_dim)
            )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts using attention mechanism

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]

        Returns:
            router_logits: [batch_size, seq_len, num_experts]
            router_weights: [batch_size, seq_len, num_experts]
        """

        batch_size, seq_len, hidden_dim = hidden_states.shape

        if self.config.enabled and hasattr(self, 'attention'):
            # Use attention-based routing
            # Expand hidden states for attention
            query = hidden_states.unsqueeze(2)  # [B, S, 1, D]

            # Expand expert embeddings
            key = self.expert_embeddings.expand(batch_size, -1, seq_len, -1)
            value = key

            # Compute attention scores
            attn_output, _ = self.attention(query, key, value, causal=False)

            # Project to router logits
            router_logits = self.router_proj(attn_output.squeeze(2))
        else:
            # Standard linear routing
            router_logits = self.router_proj(hidden_states)

        # Compute weights
        router_weights = F.softmax(router_logits, dim=-1)

        return router_logits, router_weights


def create_flash_attention(
    hidden_dim: int,
    num_heads: int,
    dropout: float = 0.0,
    causal: bool = True,
    enable: bool = True
) -> FlashAttention:
    """
    Factory function to create Flash Attention module

    Args:
        hidden_dim: Model hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        causal: Whether to use causal masking
        enable: Whether to enable Flash Attention

    Returns:
        FlashAttention module
    """

    config = FlashAttentionConfig(
        enabled=enable,
        causal=causal,
        dropout_p=dropout,
        scale=1.0 / math.sqrt(hidden_dim // num_heads)
    )

    return FlashAttention(config)


def benchmark_flash_attention():
    """
    Benchmark Flash Attention vs Standard Attention
    """

    import time

    # Test configuration
    batch_size = 4
    seq_len = 2048
    num_heads = 32
    head_dim = 96

    # Create tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)

    # Create modules
    flash_config = FlashAttentionConfig(enabled=True)
    flash_attn = FlashAttention(flash_config)

    standard_config = FlashAttentionConfig(enabled=False)
    standard_attn = FlashAttention(standard_config)

    # Warmup
    for _ in range(5):
        _ = flash_attn(q, k, v, causal=True)
        _ = standard_attn(q, k, v, causal=True)

    # Benchmark Flash Attention
    if flash_attn.flash_available:
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(20):
            output_flash, _ = flash_attn(q, k, v, causal=True)

        torch.cuda.synchronize()
        flash_time = time.time() - start

        print(f"Flash Attention: {flash_time:.3f}s")
    else:
        print("Flash Attention not available")
        flash_time = float('inf')

    # Benchmark Standard Attention
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(20):
        output_standard, _ = standard_attn(q, k, v, causal=True)

    torch.cuda.synchronize()
    standard_time = time.time() - start

    print(f"Standard Attention: {standard_time:.3f}s")

    if flash_time < float('inf'):
        speedup = standard_time / flash_time
        print(f"Speedup: {speedup:.2f}×")

    # Check memory usage
    if torch.cuda.is_available():
        print(f"\nMemory usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Flash Attention v2 Implementation")
    print("=" * 60)

    # Check hardware support
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        print(f"GPU: {name}")
        print(f"Compute Capability: {capability[0]}.{capability[1]}")

        if capability[0] >= 8:
            print("✓ Hardware supports Flash Attention v2")
        else:
            print("✗ Hardware does not support Flash Attention v2")
            print("  Ampere (RTX 30xx) or newer required")

    else:
        print("No CUDA device available")

    print("\nRunning benchmark...")
    print("-" * 60)
    benchmark_flash_attention()