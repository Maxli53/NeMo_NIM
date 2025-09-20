#!/usr/bin/env python3
"""
Expert Mixer for MoE
Handles mixing of expert outputs with proper weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ExpertMixer(nn.Module):
    """
    Mixes expert outputs according to gate weights
    """

    def __init__(self, hidden_dim: int, device: str = "cuda"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device

    def mix_expert_outputs(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_outputs: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Mix expert outputs weighted by gate scores

        Args:
            hidden_states: [batch, seq, hidden] - original input
            expert_indices: [batch, seq, k] - selected expert indices
            expert_weights: [batch, seq, k] - normalized weights
            expert_outputs: Dict mapping expert_idx to output tensor

        Returns:
            Mixed output [batch, seq, hidden]
        """
        batch_size, seq_len, _ = hidden_states.shape
        k = expert_indices.shape[-1]

        # Initialize output tensor
        output = torch.zeros_like(hidden_states)

        # Process each position
        for b in range(batch_size):
            for s in range(seq_len):
                for e in range(k):
                    expert_idx = expert_indices[b, s, e].item()
                    weight = expert_weights[b, s, e]

                    if expert_idx in expert_outputs:
                        # Apply weighted expert output
                        expert_out = expert_outputs[expert_idx]
                        # Handle batch size mismatch
                        if expert_out.shape[0] <= b:
                            # Expert output has smaller batch, use last available
                            effective_b = min(b, expert_out.shape[0] - 1)
                            output[b, s] += weight * expert_out[effective_b, s]
                        else:
                            output[b, s] += weight * expert_out[b, s]
                    else:
                        # Fallback: pass through input with weight
                        output[b, s] += weight * hidden_states[b, s]

        return output

    def mix_expert_outputs_optimized(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_outputs: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Optimized version using tensor operations

        Args:
            hidden_states: [batch, seq, hidden]
            expert_indices: [batch, seq, k]
            expert_weights: [batch, seq, k]
            expert_outputs: Dict[expert_idx -> tensor]

        Returns:
            Mixed output [batch, seq, hidden]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        k = expert_indices.shape[-1]
        device = hidden_states.device

        # Initialize output
        output = torch.zeros_like(hidden_states)

        # Create expert output tensor
        all_expert_outputs = torch.zeros(
            batch_size, seq_len, k, hidden_dim,
            dtype=hidden_states.dtype, device=device
        )

        # Gather expert outputs
        for expert_idx, expert_output in expert_outputs.items():
            # Find where this expert is selected
            mask = (expert_indices == expert_idx)
            positions = torch.where(mask)

            if len(positions[0]) > 0:
                # Assign expert output to corresponding positions
                all_expert_outputs[positions] = expert_output[
                    positions[0], positions[1]
                ]

        # Apply weights and sum
        weighted_outputs = all_expert_outputs * expert_weights.unsqueeze(-1)
        output = weighted_outputs.sum(dim=2)

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_outputs: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass"""
        return self.mix_expert_outputs(
            hidden_states, expert_indices, expert_weights, expert_outputs
        )