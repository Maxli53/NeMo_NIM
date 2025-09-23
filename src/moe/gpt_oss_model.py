#!/usr/bin/env python3
"""
Complete GPT-OSS-20B Model Implementation
Integrates attention, MoE, normalization, and embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import logging
from typing import Optional, Tuple, Dict
from safetensors import safe_open

from .native_moe_loader_v2 import GPTOSSNativeMoE
from .attention import GroupedQueryAttention
from .normalization import RMSNorm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """Single transformer block with attention and MoE"""

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        moe_handler: GPTOSSNativeMoE,
        sliding_window: int = 128,
        rope_theta: float = 150000.0,
        rope_scaling_factor: float = 32.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size

        # Layer components
        self.attn_norm = RMSNorm(hidden_size)
        self.attention = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sliding_window=sliding_window,
            rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
        )
        self.mlp_norm = RMSNorm(hidden_size)
        self.moe_handler = moe_handler  # Shared MoE handler

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        # MoE with residual
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.moe_handler.forward_layer(hidden_states, self.layer_idx)
        hidden_states = residual + hidden_states

        return hidden_states

    def load_from_safetensors(self, f, layer_idx: int):
        """Load weights from safetensors file"""
        # Load attention norm
        attn_norm_key = f"block.{layer_idx}.attn.norm.scale"
        if attn_norm_key in f.keys():
            self.attn_norm.load_from_state_dict(f.get_tensor(attn_norm_key).cuda())

        # Load attention weights
        attn_state_dict = {}
        attn_qkv_weight = f"block.{layer_idx}.attn.qkv.weight"
        attn_qkv_bias = f"block.{layer_idx}.attn.qkv.bias"
        attn_out_weight = f"block.{layer_idx}.attn.out.weight"
        attn_out_bias = f"block.{layer_idx}.attn.out.bias"
        attn_sinks = f"block.{layer_idx}.attn.sinks"

        if attn_qkv_weight in f.keys():
            attn_state_dict["qkv.weight"] = f.get_tensor(attn_qkv_weight).cuda()
        if attn_qkv_bias in f.keys():
            attn_state_dict["qkv.bias"] = f.get_tensor(attn_qkv_bias).cuda()
        if attn_out_weight in f.keys():
            attn_state_dict["out.weight"] = f.get_tensor(attn_out_weight).cuda()
        if attn_out_bias in f.keys():
            attn_state_dict["out.bias"] = f.get_tensor(attn_out_bias).cuda()
        if attn_sinks in f.keys():
            attn_state_dict["sinks"] = f.get_tensor(attn_sinks).cuda()

        self.attention.load_from_state_dict(attn_state_dict)

        # Load MLP norm
        mlp_norm_key = f"block.{layer_idx}.mlp.norm.scale"
        if mlp_norm_key in f.keys():
            self.mlp_norm.load_from_state_dict(f.get_tensor(mlp_norm_key).cuda())


class GPTOSSModel(nn.Module):
    """Complete GPT-OSS-20B Model"""

    def __init__(self, model_path: str = "gpt-oss-20b/original"):
        super().__init__()
        self.model_path = Path(model_path)

        # Load config
        with open(self.model_path / "config.json") as f:
            self.config = json.load(f)

        # Model parameters
        self.vocab_size = self.config["vocab_size"]
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_hidden_layers"]
        self.num_heads = self.config["num_attention_heads"]
        self.num_kv_heads = self.config["num_key_value_heads"]
        self.head_dim = self.config["head_dim"]
        self.sliding_window = self.config.get("sliding_window", 128)
        self.rope_theta = self.config.get("rope_theta", 150000.0)
        self.rope_scaling_factor = self.config.get("rope_scaling_factor", 32.0)

        logger.info(f"Initializing GPT-OSS-20B with {self.num_layers} layers, {self.hidden_size} hidden size")

        # Initialize MoE handler (shared across layers)
        self.moe_handler = GPTOSSNativeMoE(str(self.model_path), cache_size_gb=5.0)

        # Embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(0.0)  # No dropout for inference

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                layer_idx=i,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                moe_handler=self.moe_handler,
                sliding_window=self.sliding_window,
                rope_theta=self.rope_theta,
                rope_scaling_factor=self.rope_scaling_factor,
            )
            for i in range(self.num_layers)
        ])

        # Output
        self.ln_f = RMSNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # Load weights
        self._load_weights()

    def _load_weights(self):
        """Load all model weights from safetensors"""
        weights_path = self.model_path / "model.safetensors"

        logger.info(f"Loading weights from {weights_path}")

        with safe_open(weights_path, framework="pt", device="cpu") as f:
            # Load embeddings
            if "embedding.weight" in f.keys():
                self.embedding.weight.data = f.get_tensor("embedding.weight").cuda()
                logger.info(f"Loaded embedding weights: {self.embedding.weight.shape}")

            # Load transformer blocks
            for i, block in enumerate(self.blocks):
                block.load_from_safetensors(f, i)
                logger.debug(f"Loaded block {i}")

            # Load final norm (it's called "norm.scale" not "ln_f.scale")
            if "norm.scale" in f.keys():
                self.ln_f.load_from_state_dict(f.get_tensor("norm.scale").cuda())
                logger.info(f"Loaded final norm: {self.ln_f.scale.shape}")

            # Load LM head (might share weights with embedding)
            if "unembedding.weight" in f.keys():
                self.lm_head.weight.data = f.get_tensor("unembedding.weight").cuda()
                logger.info(f"Loaded unembedding weights: {self.lm_head.weight.shape}")
            elif "lm_head.weight" in f.keys():
                self.lm_head.weight.data = f.get_tensor("lm_head.weight").cuda()
            else:
                # Weight tying with embeddings
                self.lm_head.weight = self.embedding.weight
                logger.info("Using tied embeddings for lm_head")

        logger.info("✅ All weights loaded successfully")

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through the model

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs for RoPE
            return_dict: Whether to return a dict (for compatibility)

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        hidden_states = self.embedding(input_ids)
        hidden_states = self.dropout(hidden_states)

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, 1, seq_len, seq_len), device=input_ids.device)

        # Pass through transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask, position_ids)

        # Final norm and output projection
        hidden_states = self.ln_f(hidden_states)
        # Ensure dtype compatibility for lm_head
        hidden_states = hidden_states.to(self.lm_head.weight.dtype)
        logits = self.lm_head(hidden_states)

        if return_dict:
            return {"logits": logits}
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_context_length: int = 2048,
    ) -> torch.LongTensor:
        """
        Memory-safe text generation with sliding window

        Args:
            input_ids: Initial input token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            max_context_length: Maximum context length to keep in memory

        Returns:
            Generated token IDs
        """
        self.eval()

        # Check initial memory
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
            memory_limit = 20e9  # 20GB safety limit

        generated_tokens = []

        for i in range(max_new_tokens):
            # Memory management: clear cache periodically
            if i > 0 and i % 10 == 0:
                torch.cuda.empty_cache()

            # Check memory usage
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
                if current_memory > memory_limit:
                    logger.warning(f"Memory usage high: {current_memory/1e9:.2f}GB, clearing cache")
                    torch.cuda.empty_cache()
                    # Force clear expert cache if needed
                    if hasattr(self, 'moe_handler'):
                        self.moe_handler.expert_cache.clear()

            # Sliding window: only use last max_context_length tokens
            if input_ids.shape[1] > max_context_length:
                context_ids = input_ids[:, -max_context_length:]
                logger.debug(f"Truncating context from {input_ids.shape[1]} to {max_context_length} tokens")
            else:
                context_ids = input_ids

            # Forward pass with context window
            try:
                outputs = self.forward(context_ids)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs

                # Get logits for the last position
                next_token_logits = logits[:, -1, :].contiguous()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error("CUDA out of memory during generation, clearing cache")
                    torch.cuda.empty_cache()
                    if hasattr(self, 'moe_handler'):
                        self.moe_handler.expert_cache.clear()
                    # Try with smaller context
                    context_ids = context_ids[:, -min(128, context_ids.shape[1]):]
                    outputs = self.forward(context_ids)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    next_token_logits = logits[:, -1, :].contiguous()
                else:
                    raise e

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float("inf")

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Store generated token
            generated_tokens.append(next_token)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Clean up intermediate tensors
            del logits, next_token_logits, probs
            if 'sorted_logits' in locals():
                del sorted_logits, sorted_indices, cumulative_probs

        return input_ids

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        stats = self.moe_handler.get_memory_stats()
        stats["model_parameters"] = sum(p.numel() for p in self.parameters()) / 1e9
        return stats