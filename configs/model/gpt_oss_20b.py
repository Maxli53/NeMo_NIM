"""
GPT-OSS-20B Model Configuration
Following official NeMo documentation: https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/gpt_oss.html
"""

from nemo.collections import llm
from nemo.collections.llm.gpt_oss import GPTOSSConfig20B

# Model configuration based on official NeMo GPTOSSConfig20B
model_config = GPTOSSConfig20B()

# Model architecture details from OpenAI specification
model_params = {
    "num_layers": 44,
    "hidden_size": 6144,
    "num_attention_heads": 64,
    "seq_length": 128000,  # 128k context length
    "num_experts": 8,      # MoE configuration
    "num_experts_per_tok": 2,  # 2 experts active per token
    "ffn_hidden_size": 16384,
    "activation": "swiglu",
    "position_embedding_type": "rope",
    "vocab_size": 200064,
    "make_vocab_size_divisible_by": 128,
    "layernorm_epsilon": 1e-5,
    "init_method_std": 0.01,
    "use_cpu_initialization": False,
    "seed": 1234,
}

# Training optimization parameters
optimization_params = {
    "fp16": False,
    "bf16": True,  # Use BF16 for training
    "params_dtype": "bf16",
    "pipeline_dtype": "bf16",
    "autocast_dtype": "bf16",
    "grad_scaler": False,  # Not needed with bf16
}

# Memory optimization for RTX 3090 (24GB)
memory_optimization = {
    "activations_checkpoint_method": "uniform",
    "activations_checkpoint_num_layers": 1,
    "sequence_parallel": True,
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "virtual_pipeline_model_parallel_size": None,
    "context_parallel_size": 1,
    "expert_model_parallel_size": 1,
    "use_flash_attention": True,
}