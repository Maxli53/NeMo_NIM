#!/usr/bin/env python3
"""
Attention implementation for GPT-OSS-20B
Implements Grouped Query Attention (GQA) with RoPE embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 150000.0,  # GPT-OSS uses 150000
        scaling_factor: float = 32.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor

        # Compute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build cache for positions
        self._build_cache()

    def _build_cache(self):
        max_seq_len_cached = self.max_position_embeddings
        t = torch.arange(max_seq_len_cached, dtype=torch.float32)
        t = t / self.scaling_factor  # Apply scaling

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch_size, num_heads, seq_len, head_dim]
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention for GPT-OSS-20B
    - 64 query heads
    - 8 key-value heads (8x fewer than Q heads)
    - Sliding window attention (128 tokens)
    """

    def __init__(
        self,
        hidden_size: int = 2880,
        num_heads: int = 64,
        num_kv_heads: int = 8,
        head_dim: int = 64,
        sliding_window: int = 128,
        rope_theta: float = 150000.0,
        rope_scaling_factor: float = 32.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window

        # Validate dimensions
        assert hidden_size % num_heads == 0
        self.num_groups = num_heads // num_kv_heads

        # QKV projections
        # Q: [hidden_size, num_heads * head_dim] = [2880, 4096]
        # KV: [hidden_size, 2 * num_kv_heads * head_dim] = [2880, 1024]
        self.qkv = nn.Linear(hidden_size, (num_heads + 2 * num_kv_heads) * head_dim, bias=True)
        self.out = nn.Linear(num_heads * head_dim, hidden_size, bias=True)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            head_dim,
            base=rope_theta,
            scaling_factor=rope_scaling_factor,
        )

        # Attention sinks (special tokens)
        self.register_buffer("sinks", torch.zeros(num_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection and split
        qkv = self.qkv(hidden_states)

        # Split into Q, K, V
        q_size = self.num_heads * self.head_dim
        k_size = self.num_kv_heads * self.head_dim
        v_size = self.num_kv_heads * self.head_dim

        q, k, v = torch.split(qkv, [q_size, k_size, v_size], dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(q, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat KV heads for each group
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # Apply sliding window attention if needed
        if self.sliding_window is not None and seq_len > self.sliding_window:
            # Create sliding window mask
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, 1, seq_len, seq_len), device=hidden_states.device)

            # Apply sliding window
            for i in range(seq_len):
                start = max(0, i - self.sliding_window + 1)
                if start > 0:
                    attention_mask[:, :, i, :start] = 0

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + (1.0 - attention_mask) * -10000.0

        # Apply attention sinks (stabilization technique)
        scores[:, :, :, 0] = scores[:, :, :, 0] + self.sinks.view(1, -1, 1)

        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.out(attn_output)

        return output

    def load_from_state_dict(self, state_dict: dict):
        """Load pretrained weights"""
        if "qkv.weight" in state_dict:
            self.qkv.weight.data = state_dict["qkv.weight"]
        if "qkv.bias" in state_dict:
            self.qkv.bias.data = state_dict["qkv.bias"]
        if "out.weight" in state_dict:
            self.out.weight.data = state_dict["out.weight"]
        if "out.bias" in state_dict:
            self.out.bias.data = state_dict["out.bias"]
        if "sinks" in state_dict:
            self.sinks.data = state_dict["sinks"]