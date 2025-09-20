#!/usr/bin/env python3
"""
MXFP4 Handler for GPT-OSS-20B
Handles 4-bit mantissa with shared exponent quantization
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def swiglu(x: torch.Tensor) -> torch.Tensor:
    """
    SwiGLU activation: gate * silu(up)

    Args:
        x: Input tensor where last dim contains [gate, up] concatenated

    Returns:
        Activated output
    """
    gate, up = x.chunk(2, dim=-1)
    return gate * F.silu(up)


class MXFP4Handler:
    """
    Handles MXFP4 quantization/dequantization for GPT-OSS experts
    """

    @staticmethod
    def dequantize_mxfp4(
        blocks: torch.Tensor,
        scales: torch.Tensor,
        block_size: int = 32
    ) -> torch.Tensor:
        """
        Dequantize MXFP4 format to bfloat16

        Args:
            blocks: Quantized blocks (int8 format storing 4-bit values)
            scales: Per-block scales (bfloat16)
            block_size: Size of quantization blocks

        Returns:
            Dequantized tensor in bfloat16
        """
        # Convert int8 blocks to bfloat16 for computation
        blocks_bf16 = blocks.to(torch.bfloat16)

        # Apply scales
        if scales.dim() == blocks.dim() - 1:
            scales = scales.unsqueeze(-1)

        # Dequantize
        dequantized = blocks_bf16 * scales

        return dequantized

    @staticmethod
    def apply_expert_ffn(
        hidden_states: torch.Tensor,
        gate_up_blocks: torch.Tensor,
        gate_up_scales: torch.Tensor,
        gate_up_bias: Optional[torch.Tensor],
        down_blocks: torch.Tensor,
        down_scales: torch.Tensor,
        down_bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply expert FFN with MXFP4 weights

        Args:
            hidden_states: Input [batch, seq, hidden]
            gate_up_blocks: Gate-up projection blocks
            gate_up_scales: Gate-up scales
            gate_up_bias: Optional gate-up bias
            down_blocks: Down projection blocks
            down_scales: Down scales
            down_bias: Optional down bias

        Returns:
            Expert output [batch, seq, hidden]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Dequantize gate-up weights
        gate_up_weight = MXFP4Handler.dequantize_mxfp4(
            gate_up_blocks, gate_up_scales
        )

        # Reshape for linear projection
        # Expected shape: [hidden_dim, ffn_dim * 2]
        if gate_up_weight.dim() > 2:
            gate_up_weight = gate_up_weight.view(hidden_dim, -1)

        # Gate-up projection
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        gate_up_output = F.linear(hidden_states_flat, gate_up_weight.T, gate_up_bias)

        # Apply SwiGLU activation
        activated = swiglu(gate_up_output)

        # Dequantize down weights
        down_weight = MXFP4Handler.dequantize_mxfp4(down_blocks, down_scales)

        # Reshape for linear projection
        if down_weight.dim() > 2:
            down_weight = down_weight.view(-1, hidden_dim)

        # Down projection
        output = F.linear(activated, down_weight.T, down_bias)

        # Reshape back
        output = output.view(batch_size, seq_len, hidden_dim)

        return output

    @staticmethod
    def quantize_to_mxfp4(
        tensor: torch.Tensor,
        block_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to MXFP4 format

        Args:
            tensor: Input tensor to quantize
            block_size: Size of quantization blocks

        Returns:
            (blocks, scales) tuple
        """
        # Flatten tensor for processing
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()

        # Pad if necessary
        pad_size = (block_size - len(tensor_flat) % block_size) % block_size
        if pad_size > 0:
            tensor_flat = F.pad(tensor_flat, (0, pad_size))

        # Reshape into blocks
        tensor_blocks = tensor_flat.view(-1, block_size)

        # Compute per-block scales
        scales = tensor_blocks.abs().max(dim=1)[0] / 7  # 4-bit signed range

        # Avoid division by zero
        scales = torch.clamp(scales, min=1e-8)

        # Quantize
        blocks_quantized = torch.round(tensor_blocks / scales.unsqueeze(1))
        blocks_quantized = torch.clamp(blocks_quantized, -8, 7)

        # Convert to int8
        blocks_int8 = blocks_quantized.to(torch.int8)

        return blocks_int8, scales.to(torch.bfloat16)