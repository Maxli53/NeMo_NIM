#!/usr/bin/env python3
"""
GPT-OSS-20B Inference Script - Comprehensive Configuration
All configurable parameters with beginner-friendly explanations
"""

from nemo.collections import llm
from nemo.collections.llm import api
import nemo.lightning as nl
from megatron.core.inference.common_inference_params import CommonInferenceParams
import torch

# ============================================================================
# SECTION 1: CRITICAL CONFIGS (START HERE - Most commonly adjusted)
# ============================================================================

MODEL_PATH = "/workspace/checkpoints/gpt-oss-20b"

# MoE Top-K: Number of experts to activate per token (MOST IMPORTANT setting)
#   - 1 = Fastest, lowest quality, ~8GB VRAM
#   - 2 = Very fast, decent quality, ~10GB VRAM ✓ RECOMMENDED for 24GB GPU
#   - 4 = Balanced, good quality, ~16GB VRAM
#   - 8 = Slower, better quality, ~20GB VRAM
#   - 16 = Much slower, high quality, ~24GB VRAM (near limit)
#   - 32 = All experts, maximum quality, may exceed VRAM
#   Impact: Higher = Better quality but slower and more VRAM
MOE_TOPK = 2

# Temperature: Controls randomness in text generation
#   - 0.0 = Deterministic (always picks most likely token)
#   - 0.3 = Focused, coherent (good for factual tasks)
#   - 0.7 = Balanced creativity (RECOMMENDED for general use)
#   - 1.0 = Creative, varied outputs
#   - 1.5+ = Very random, may be incoherent
#   Impact: Higher = More creative but less predictable
TEMPERATURE = 0.7

# Top-P (Nucleus Sampling): Consider only top tokens with cumulative probability P
#   - 0.1 = Very focused, predictable
#   - 0.5 = Moderately focused
#   - 0.9 = Balanced (RECOMMENDED)
#   - 0.95 = More variety
#   - 1.0 = Consider all tokens
#   Impact: Lower = More focused, higher = more diverse
TOP_P = 0.9

# Top-K: Consider only the top K most likely tokens
#   - 1 = Always pick most likely (greedy)
#   - 10 = Very focused
#   - 50 = Balanced (RECOMMENDED)
#   - 100+ = More variety
#   - 0 = Disabled (use top_p instead)
#   Impact: Lower = More focused, higher = more diverse
TOP_K = 50

# Number of Tokens to Generate: Maximum output length
#   - 50 = Short response (1-2 sentences)
#   - 100 = Medium response (paragraph)
#   - 256 = Long response (multiple paragraphs)
#   - 512 = Very long response
#   Impact: Higher = Longer output but slower generation
NUM_TOKENS_TO_GENERATE = 100

# Your prompts to generate from
PROMPTS = [
    "Q: What is artificial intelligence?",
    "Explain quantum computing in simple terms:",
    "Write a short story about a robot:",
]

# ============================================================================
# SECTION 2: MoE ADVANCED SETTINGS (Mixture of Experts - 35+ parameters)
# ============================================================================

# Load Balancing: How to distribute tokens across experts
#   - "none" = No load balancing (DEFAULT, fastest)
#   - "aux_loss" = Auxiliary loss for balance (improves distribution)
#   - "sinkhorn" = Sinkhorn algorithm (more complex balancing)
#   Impact: Helps prevent some experts being overused while others idle
MOE_LOAD_BALANCING_TYPE = "none"

# Auxiliary Loss Coefficient: Weight for load balancing loss (if enabled)
#   - 0.0 = No auxiliary loss (DEFAULT)
#   - 0.01 = Light balancing (typical)
#   - 0.1 = Strong balancing
#   Impact: Only used if MOE_LOAD_BALANCING_TYPE is "aux_loss"
MOE_AUX_LOSS_COEFF = 0.0

# Token Dispatcher: How tokens are routed to experts across GPUs
#   - "alltoall" = All-to-all communication (DEFAULT, efficient)
#   - "allgather" = All-gather communication (alternative method)
#   Impact: Different communication patterns for multi-GPU setups
MOE_TOKEN_DISPATCHER_TYPE = "alltoall"

# Grouped GEMM: Use grouped matrix multiplication for experts
#   - True = Use optimized grouped operations (DEFAULT, RECOMMENDED)
#   - False = Use separate operations (slower)
#   Impact: Significant speedup when enabled
MOE_GROUPED_GEMM = True

# Permute Fusion: Fuse permutation operations with compute
#   - True = Fuse operations (DEFAULT, RECOMMENDED)
#   - False = Separate operations
#   Impact: Minor speedup, reduces kernel launches
MOE_PERMUTE_FUSION = True

# Pre-Softmax: Apply softmax before or after routing
#   - False = Post-softmax (DEFAULT)
#   - True = Pre-softmax
#   Impact: Affects numerical stability of routing
MOE_ROUTER_PRE_SOFTMAX = False

# Router Score Function: How expert scores are computed
#   - "softmax" = Standard softmax (DEFAULT)
#   - "sigmoid" = Sigmoid activation (alternative)
#   Impact: Changes how routing probabilities are calculated
MOE_ROUTER_SCORE_FUNCTION = "softmax"

# Expert Capacity Factor: Limit tokens per expert
#   - None = No capacity limit (DEFAULT)
#   - 1.0 = Each expert can handle avg number of tokens
#   - 1.5 = 50% buffer capacity
#   Impact: Prevents expert overload but may drop tokens
MOE_EXPERT_CAPACITY_FACTOR = None

# Token Dropping: Drop tokens if expert capacity exceeded
#   - False = Don't drop tokens (DEFAULT)
#   - True = Drop overflow tokens
#   Impact: Only relevant if capacity factor is set
MOE_TOKEN_DROPPING = False

# Token Drop Policy: Which tokens to drop if dropping is enabled
#   - "probs" = Drop based on routing probabilities (DEFAULT)
#   - "position" = Drop based on position
#   Impact: Only relevant if token dropping is enabled
MOE_TOKEN_DROP_POLICY = "probs"

# Input Jitter: Add noise to router inputs for regularization
#   - None = No jitter (DEFAULT)
#   - 0.01 = Light jitter (typical value if used)
#   Impact: Can help with training, usually disabled for inference
MOE_INPUT_JITTER_EPS = None

# Z-Loss Coefficient: Regularization for router logits
#   - None = No z-loss (DEFAULT for inference)
#   - 0.001 = Light regularization (training)
#   Impact: Typically used during training, not inference
MOE_Z_LOSS_COEFF = None

# Router Fusion: Fuse router computation with other ops
#   - False = No fusion (DEFAULT)
#   - True = Fuse operations
#   Impact: Potential speedup, experimental feature
MOE_ROUTER_FUSION = False

# FFN Hidden Size: Hidden dimension of expert feed-forward networks
#   - 2880 = DEFAULT for GPT-OSS-20B (matches architecture)
#   Impact: DO NOT CHANGE (baked into checkpoint)
MOE_FFN_HIDDEN_SIZE = 2880

# Number of Experts: Total experts in the model
#   - 32 = GPT-OSS-20B has 32 experts
#   Impact: DO NOT CHANGE (baked into checkpoint)
NUM_MOE_EXPERTS = 32

# Layer Frequency: Apply MoE every N layers
#   - 1 = Every layer has MoE (DEFAULT)
#   - 2 = Every other layer
#   Impact: DO NOT CHANGE (baked into checkpoint)
MOE_LAYER_FREQ = 1

# Additional Advanced MoE Settings (rarely changed):
MOE_APPLY_PROBS_ON_INPUT = False
MOE_DEEPEP_NUM_SMS = 20
MOE_ENABLE_DEEPEP = False
MOE_EXTENDED_TP = False
MOE_LAYER_RECOMPUTE = False
MOE_PAD_EXPERT_INPUT_TO_CAPACITY = False
MOE_PER_LAYER_LOGGING = False
MOE_ROUTER_BIAS_UPDATE_RATE = 0.001
MOE_ROUTER_DTYPE = None
MOE_ROUTER_ENABLE_EXPERT_BIAS = False
MOE_ROUTER_FORCE_LOAD_BALANCING = False
MOE_ROUTER_GROUP_TOPK = None
MOE_ROUTER_NUM_GROUPS = None
MOE_ROUTER_PADDING_FOR_FP8 = False
MOE_ROUTER_TOPK_LIMITED_DEVICES = None
MOE_ROUTER_TOPK_SCALING_FACTOR = None
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE = None
MOE_SHARED_EXPERT_OVERLAP = False
MOE_USE_LEGACY_GROUPED_GEMM = False
OVERLAP_MOE_EXPERT_PARALLEL_COMM = False

# ============================================================================
# SECTION 3: PARALLELISM (Multi-GPU distribution)
# ============================================================================

# Tensor Parallel: Split model layers across N GPUs
#   - 1 = No tensor parallelism (DEFAULT for single GPU)
#   - 2 = Split across 2 GPUs
#   - 4/8 = Split across 4/8 GPUs
#   Impact: Enables larger models on multiple GPUs
TENSOR_PARALLEL_SIZE = 1

# Pipeline Parallel: Split model depth across N GPUs
#   - 1 = No pipeline parallelism (DEFAULT)
#   - 2+ = Pipeline stages across GPUs
#   Impact: Different from tensor parallel, splits by layers
PIPELINE_PARALLEL_SIZE = 1

# Expert Parallel: Distribute experts across N GPUs
#   - 1 = No expert parallelism (DEFAULT)
#   - 2/4/8 = Distribute 32 experts across GPUs
#   Impact: MoE-specific parallelism for multi-GPU
EXPERT_PARALLEL_SIZE = 1

# Expert Tensor Parallel: Tensor parallel within each expert
#   - None = Use same as TENSOR_PARALLEL_SIZE (DEFAULT)
#   - 1 = Disable expert TP (recommended with expert parallel)
#   Impact: Advanced setting for large-scale deployments
EXPERT_TENSOR_PARALLEL_SIZE = None

# Context Parallel: Split sequence length across GPUs
#   - 1 = No context parallelism (DEFAULT)
#   - 2+ = Split long sequences across GPUs
#   Impact: For very long sequences (>128k tokens)
CONTEXT_PARALLEL_SIZE = 1

# Sequence Parallel: Additional parallelism for sequences
#   - False = Disabled (DEFAULT)
#   - True = Enable sequence parallelism
#   Impact: Works with tensor parallel for efficiency
SEQUENCE_PARALLEL = False

# Number of GPUs to use on this node
#   - 1 = Single GPU (RTX 3090)
#   - 2/4/8 = Multiple GPUs if available
DEVICES = 1

# Number of nodes (multi-node training)
#   - 1 = Single machine (DEFAULT)
#   - 2+ = Multi-node cluster
NUM_NODES = 1

# ============================================================================
# SECTION 4: PRECISION & QUANTIZATION (Memory optimization)
# ============================================================================

# Precision: Floating point precision for computation
#   - "bf16-mixed" = BFloat16 mixed precision (RECOMMENDED, best balance)
#   - "16-mixed" = FP16 mixed precision (slightly faster, less stable)
#   - "32" = Full FP32 (highest precision, 2x memory, slower)
#   Impact: Lower precision = faster + less VRAM but slightly less accurate
PRECISION = "bf16-mixed"

# FP8 Inference: Use 8-bit floating point (experimental)
#   - False = Disabled (DEFAULT, RECOMMENDED)
#   - True = Enable FP8 (requires special hardware support)
#   Impact: Significant VRAM savings but requires Hopper+ GPUs
FP8_ENABLED = False

# FP8 Recipe: How to apply FP8 quantization (if enabled)
#   - "tensorwise" = Per-tensor scaling (DEFAULT)
#   - "delayed" = Delayed scaling
#   - "mxfp8" = Microscaling FP8
#   Impact: Only relevant if FP8_ENABLED = True
FP8_RECIPE = "tensorwise"

# FP8 AMAX History: History length for FP8 scaling
#   - 1 = Use only current batch (DEFAULT)
#   - 16/32 = Use moving average
#   Impact: Only relevant if FP8_ENABLED = True
FP8_AMAX_HISTORY_LEN = 1

# FP8 AMAX Algorithm: How to compute FP8 scaling factors
#   - "max" = Use maximum value (DEFAULT for inference)
#   - "most_recent" = Use most recent value
#   Impact: Only relevant if FP8_ENABLED = True
FP8_AMAX_COMPUTE_ALGO = "max"

# Autocast: Automatically cast operations to lower precision
#   - False = Manual precision control (DEFAULT)
#   - True = Auto-cast operations
#   Impact: Usually handled by precision setting above
AUTOCAST_ENABLED = False

# Gradient Reduce in FP32: Use FP32 for gradient reduction
#   - False = Use model precision (DEFAULT)
#   - True = Use FP32 (better for training, not inference)
#   Impact: Not relevant for inference
GRAD_REDUCE_IN_FP32 = False

# Use CPU Initialization: Initialize model weights on CPU first
#   - False = Initialize on GPU (DEFAULT, faster but needs VRAM)
#   - True = Initialize on CPU (CRITICAL FOR 24GB VRAM, avoids OOM)
#   Impact: Enables running GPT-OSS-20B on single 24GB GPU
#   Note: Slower initialization but prevents out-of-memory during model creation
#   IMPORTANT: Set to True for single GPU with limited VRAM
USE_CPU_INITIALIZATION = True

# ============================================================================
# SECTION 5: GENERATION PARAMETERS (Output control)
# ============================================================================

# Repetition Penalty: Penalize repeated tokens
#   - 1.0 = No penalty (DEFAULT)
#   - 1.1 = Light penalty (reduces repetition)
#   - 1.2-1.5 = Moderate penalty
#   - 2.0+ = Strong penalty (may hurt coherence)
#   Impact: Higher = Less repetitive but may sound unnatural
REPETITION_PENALTY = 1.0

# Length Penalty: Favor longer or shorter sequences
#   - 1.0 = No preference (DEFAULT)
#   - <1.0 = Favor shorter outputs
#   - >1.0 = Favor longer outputs
#   Impact: Affects beam search (not used in sampling)
LENGTH_PENALTY = 1.0

# Min Length: Minimum number of tokens to generate
#   - 0 = No minimum (DEFAULT)
#   - 10/20 = Force at least this many tokens
#   Impact: Prevents very short outputs
MIN_LENGTH = 0

# Add BOS Token: Add beginning-of-sequence token to prompt
#   - False = Don't add BOS (DEFAULT)
#   - True = Add BOS token
#   Impact: Depends on tokenizer, usually automatic
ADD_BOS = False

# Return Log Probabilities: Return token probabilities
#   - False = Just return text (DEFAULT)
#   - True = Return log probs for each token
#   Impact: Useful for analysis, adds overhead
RETURN_LOG_PROBS = False

# Top N Log Probs: Return top N token probabilities
#   - 0 = Don't return (DEFAULT)
#   - 5/10 = Return top N alternatives per token
#   Impact: Only relevant if RETURN_LOG_PROBS = True
TOP_N_LOGPROBS = 0

# ============================================================================
# SECTION 6: RUNTIME SETTINGS (Performance tuning)
# ============================================================================

# Max Batch Size: Process multiple prompts simultaneously
#   - 1 = Process one prompt at a time ✓ RECOMMENDED for 24GB GPU
#   - 4 = Process 4 prompts together
#   - 8 = Process 8 prompts together (faster for many prompts)
#   - 16+ = Even larger batches (watch VRAM usage)
#   Impact: Higher = More efficient for many prompts but more VRAM
MAX_BATCH_SIZE = 1

# Random Seed: Seed for reproducible generation
#   - 1234 = DEFAULT seed
#   - Any integer = Use specific seed for reproducibility
#   - Change value = Get different but reproducible outputs
#   Impact: Same seed + temperature=0 = identical outputs
RANDOM_SEED = 1234

# Enable Flash Decode: Use flash attention for decoding
#   - True = Enable flash decode (DEFAULT, RECOMMENDED)
#   - False = Disable (use if encountering errors)
#   Impact: Significant speedup when enabled
ENABLE_FLASH_DECODE = True

# Text Only: Return only generated text (no metadata)
#   - True = Just text (DEFAULT, RECOMMENDED)
#   - False = Return full structured output
#   Impact: Simpler output format
TEXT_ONLY = True

# Legacy Checkpoint: Load checkpoints from older Transformer Engine
#   - False = Modern checkpoint format (DEFAULT)
#   - True = Load TE < 1.14 checkpoints
#   Impact: Only needed for old checkpoints
LEGACY_CKPT = False

# ============================================================================
# IMPLEMENTATION CODE (Runs the inference with above configs)
# ============================================================================

if __name__ == "__main__":
    import os
    # Enable expandable memory segments to avoid fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    print("=" * 70)
    print("GPT-OSS-20B Inference Configuration")
    print("=" * 70)
    print(f"Model Path: {MODEL_PATH}")
    print(f"MoE Top-K: {MOE_TOPK} experts (out of {NUM_MOE_EXPERTS} total)")
    print(f"Temperature: {TEMPERATURE}, Top-P: {TOP_P}, Top-K: {TOP_K}")
    print(f"Max Tokens: {NUM_TOKENS_TO_GENERATE}")
    print(f"Precision: {PRECISION}")
    print(f"CPU Init: {USE_CPU_INITIALIZATION} (CRITICAL for 24GB VRAM)")
    print(f"Devices: {DEVICES} GPU(s)")
    print(f"Batch Size: {MAX_BATCH_SIZE}")
    print(f"Number of Prompts: {len(PROMPTS)}")
    print("=" * 70)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_model_parallel_size=PIPELINE_PARALLEL_SIZE,
        expert_model_parallel_size=EXPERT_PARALLEL_SIZE,
        expert_tensor_parallel_size=EXPERT_TENSOR_PARALLEL_SIZE,
        context_parallel_size=CONTEXT_PARALLEL_SIZE,
        sequence_parallel=SEQUENCE_PARALLEL,
        setup_optimizers=False,
        store_optimizer_states=False,
    )

    fp8_config = "hybrid" if FP8_ENABLED else None
    fp8_recipe_config = FP8_RECIPE if FP8_ENABLED else None

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=DEVICES,
        num_nodes=NUM_NODES,
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(
            precision=PRECISION,
            params_dtype=torch.bfloat16 if "bf16" in PRECISION else torch.float16,
            pipeline_dtype=torch.bfloat16 if "bf16" in PRECISION else torch.float16,
            autocast_enabled=AUTOCAST_ENABLED,
            grad_reduce_in_fp32=GRAD_REDUCE_IN_FP32,
            fp8=fp8_config,
            fp8_recipe=fp8_recipe_config,
            fp8_amax_history_len=FP8_AMAX_HISTORY_LEN,
            fp8_amax_compute_algo=FP8_AMAX_COMPUTE_ALGO,
        ),
    )

    if LEGACY_CKPT:
        trainer.strategy.ckpt_load_strictness = False

    print("\nGenerating responses...\n")

    results = api.generate(
        path=MODEL_PATH,
        prompts=PROMPTS,
        trainer=trainer,
        add_BOS=ADD_BOS,
        inference_params=CommonInferenceParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            num_tokens_to_generate=NUM_TOKENS_TO_GENERATE,
            return_log_probs=RETURN_LOG_PROBS,
            top_n_logprobs=TOP_N_LOGPROBS,
        ),
        text_only=TEXT_ONLY,
        max_batch_size=MAX_BATCH_SIZE,
        random_seed=RANDOM_SEED,
        enable_flash_decode=ENABLE_FLASH_DECODE,
        use_cpu_initialization=USE_CPU_INITIALIZATION,
    )

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        exit(0)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for i, (prompt, result) in enumerate(zip(PROMPTS, results), 1):
        print(f"\n[Prompt {i}]")
        print(f"{prompt}")
        print(f"\n[Response {i}]")
        print(f"{result}")
        print("\n" + "-" * 70)

    print("\n" + "=" * 70)
    print("Inference Complete!")
    print("=" * 70)