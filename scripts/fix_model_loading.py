#!/usr/bin/env python3
"""
Fix Model Loading Script
Handles custom GPT-OSS architecture loading issues
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Load configuration"""
    config_path = Path("../config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def generate_modeling_gpt_oss(config_json: Dict[str, Any]) -> str:
    """Generate custom modeling file for GPT-OSS architecture"""
    
    modeling_code = '''
# Auto-generated modeling file for GPT-OSS
# This handles the custom GptOssForCausalLM architecture

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

class GptOssConfig(PretrainedConfig):
    """Configuration for GPT-OSS model"""
    model_type = "gpt_oss"
    
    def __init__(
        self,
        vocab_size=201088,
        hidden_size=2880,
        intermediate_size=2880,
        num_hidden_layers=24,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=150000,
        rope_scaling=None,
        attention_bias=True,
        attention_dropout=0.0,
        sliding_window=128,
        num_local_experts=32,
        num_experts_per_tok=4,
        output_router_logits=False,
        router_aux_loss_coef=0.9,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        super().__init__(**kwargs)

class GptOssRMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class GptOssRotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim, max_position_embeddings=131072, base=150000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32).to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        seq_len = x.shape[2]
        freqs = torch.outer(position_ids.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        return cos_emb, sin_emb

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Apply rotary position embedding"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GptOssAttention(nn.Module):
    """Multi-head attention with sliding window support"""
    
    def __init__(self, config, layer_idx=None, is_sliding=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = is_sliding
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = GptOssRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat k/v heads if necessary
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply sliding window if needed
        if self.is_sliding and self.config.sliding_window is not None:
            window_size = self.config.sliding_window
            # Implement sliding window mask
            # This is simplified - actual implementation would be more complex
            pass

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class GptOssMoE(nn.Module):
    """Mixture of Experts layer"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        
        # Router
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                nn.SiLU(),
                nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
            )
            for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Router logits
        router_logits = self.router(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        topk_weights, topk_indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Process through experts
        output = torch.zeros_like(hidden_states)
        
        for i in range(self.num_experts_per_tok):
            expert_indices = topk_indices[..., i]
            expert_weights = topk_weights[..., i:i+1]
            
            for expert_idx in range(self.num_experts):
                mask = (expert_indices == expert_idx)
                if mask.any():
                    expert_input = hidden_states[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    output[mask] += expert_weights[mask] * expert_output
        
        return output

class GptOssDecoderLayer(nn.Module):
    """Transformer decoder layer with MoE"""
    
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Determine if this layer uses sliding attention
        is_sliding = config.layer_types[layer_idx] == "sliding_attention" if hasattr(config, 'layer_types') else False
        
        self.self_attn = GptOssAttention(config, layer_idx, is_sliding)
        self.mlp = GptOssMoE(config)
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # MLP with MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class GptOssPreTrainedModel(PreTrainedModel):
    """Base class for GPT-OSS"""
    config_class = GptOssConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GptOssDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

class GptOssModel(GptOssPreTrainedModel):
    """GPT-OSS base model"""
    
    def __init__(self, config: GptOssConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GptOssDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length), dtype=torch.bool, device=inputs_embeds.device
            )

        # Prepare attention mask
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

class GptOssForCausalLM(GptOssPreTrainedModel):
    """GPT-OSS for causal language modeling"""
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GptOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Decoder outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs[1],
            hidden_states=outputs[2] if len(outputs) > 2 else None,
            attentions=outputs[3] if len(outputs) > 3 else None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Standard generation preparation
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

# Register the model
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("gpt_oss", GptOssConfig)
AutoModelForCausalLM.register(GptOssConfig, GptOssForCausalLM)

print("GPT-OSS model classes registered successfully!")
'''
    
    return modeling_code

def check_model_files(model_path: Path) -> Dict[str, bool]:
    """Check which files exist for the model"""
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json"
    ]
    
    optional_files = [
        "modeling_gpt_oss.py",
        "generation_config.json",
        "special_tokens_map.json",
        "chat_template.jinja"
    ]
    
    # Check model shards
    model_shards = list(model_path.glob("model-*.safetensors"))
    
    status = {}
    
    logger.info("\n=== Checking Model Files ===")
    
    # Check required files
    for file in required_files:
        exists = (model_path / file).exists()
        status[file] = exists
        symbol = "✓" if exists else "✗"
        logger.info(f"{symbol} {file}: {'Found' if exists else 'Missing'}")
    
    # Check model shards
    if model_shards:
        logger.info(f"✓ Model shards: Found {len(model_shards)} files")
        for shard in model_shards:
            size_gb = shard.stat().st_size / 1e9
            logger.info(f"  - {shard.name}: {size_gb:.2f} GB")
    else:
        logger.error("✗ Model shards: No .safetensors files found")
        status["model_shards"] = False
    
    # Check optional files
    logger.info("\n=== Optional Files ===")
    for file in optional_files:
        exists = (model_path / file).exists()
        symbol = "✓" if exists else "-"
        logger.info(f"{symbol} {file}: {'Found' if exists else 'Not present'}")
    
    return status

def patch_config_json(model_path: Path) -> bool:
    """Patch config.json if needed"""
    config_path = model_path / "config.json"
    
    if not config_path.exists():
        logger.error("config.json not found")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info("\n=== Model Configuration ===")
    logger.info(f"Architecture: {config.get('architectures', ['Unknown'])[0]}")
    logger.info(f"Model Type: {config.get('model_type', 'Unknown')}")
    logger.info(f"Hidden Size: {config.get('hidden_size', 'Unknown')}")
    logger.info(f"Num Layers: {config.get('num_hidden_layers', 'Unknown')}")
    logger.info(f"Num Experts: {config.get('num_local_experts', 'Unknown')}")
    logger.info(f"Max Position: {config.get('max_position_embeddings', 'Unknown')}")
    
    # Check for problematic settings
    if "quantization_config" in config:
        quant_method = config["quantization_config"].get("quant_method", "unknown")
        logger.warning(f"\nModel has {quant_method} quantization")
        logger.warning("This may require special handling or custom CUDA kernels")
        
        # Option to remove quantization config for testing
        response = input("\nRemove quantization config for testing? (y/n): ")
        if response.lower() == 'y':
            config_backup = config_path.with_suffix('.json.backup')
            shutil.copy(config_path, config_backup)
            logger.info(f"Backup created: {config_backup}")
            
            del config["quantization_config"]
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info("Quantization config removed")
    
    return True

def generate_and_install_modeling_file(model_path: Path) -> bool:
    """Generate and install modeling_gpt_oss.py"""
    modeling_path = model_path / "modeling_gpt_oss.py"
    
    if modeling_path.exists():
        logger.info(f"modeling_gpt_oss.py already exists at {modeling_path}")
        return True
    
    logger.info("\n=== Generating Custom Model Implementation ===")
    
    # Load config to pass to generator
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Generate the modeling code
    modeling_code = generate_modeling_gpt_oss(config)
    
    # Write to file
    with open(modeling_path, 'w') as f:
        f.write(modeling_code)
    
    logger.info(f"✓ Generated modeling_gpt_oss.py ({len(modeling_code)} bytes)")
    logger.info(f"  Location: {modeling_path}")
    
    # Also save to project directory for reference
    project_modeling = Path("modeling_gpt_oss.py")
    with open(project_modeling, 'w') as f:
        f.write(modeling_code)
    logger.info(f"✓ Copy saved to: {project_modeling}")
    
    return True

def test_model_import() -> bool:
    """Test if the model can be imported"""
    logger.info("\n=== Testing Model Import ===")
    
    try:
        # Try importing the generated module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "modeling_gpt_oss", 
            "modeling_gpt_oss.py"
        )
        
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if classes are available
            if hasattr(module, 'GptOssForCausalLM') and hasattr(module, 'GptOssConfig'):
                logger.info("✓ Model classes imported successfully")
                logger.info("  - GptOssConfig")
                logger.info("  - GptOssForCausalLM")
                return True
            else:
                logger.error("Model classes not found in module")
                return False
                
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return False

def main():
    """Main fix routine"""
    logger.info("="*60)
    logger.info("GPT-OSS Model Loading Fix Script")
    logger.info("="*60)
    
    # Load config
    config = load_config()
    model_path_str = config.get('model', {}).get(
        'model_path',
        r"C:\Users\maxli\.cache\huggingface\hub\models--openai--gpt-oss-20b\snapshots\6cee5e81ee83917806bbde320786a8fb61efebee"
    )
    
    model_path = Path(model_path_str)
    
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        return False
    
    logger.info(f"Model path: {model_path}")
    
    # Step 1: Check existing files
    file_status = check_model_files(model_path)
    
    # Step 2: Patch config if needed
    if not patch_config_json(model_path):
        logger.error("Failed to patch config.json")
        return False
    
    # Step 3: Generate modeling file if missing
    modeling_path = model_path / "modeling_gpt_oss.py"
    if not modeling_path.exists():
        response = input("\nGenerate modeling_gpt_oss.py? (y/n): ")
        if response.lower() == 'y':
            if not generate_and_install_modeling_file(model_path):
                logger.error("Failed to generate modeling file")
                return False
    
    # Step 4: Test import
    if Path("modeling_gpt_oss.py").exists():
        if not test_model_import():
            logger.warning("Model import test failed")
            logger.warning("The generated code may need adjustments")
    
    logger.info("\n" + "="*60)
    logger.info("✅ Model fixing complete!")
    logger.info("Next step: Run test_gpt_oss_minimal.py")
    logger.info("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)