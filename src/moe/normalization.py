#!/usr/bin/env python3
"""
RMSNorm implementation for GPT-OSS-20B
RMSNorm is used instead of LayerNorm in the model
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Normalization

    Used in GPT-OSS-20B instead of LayerNorm for better efficiency
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Normalized tensor with same shape as input
        """
        # Calculate RMS
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x * torch.rsqrt(norm + self.eps)

        # Apply learned scale
        return x_normalized * self.scale

    def load_from_state_dict(self, scale_tensor: torch.Tensor):
        """Load pretrained scale weights"""
        self.scale.data = scale_tensor