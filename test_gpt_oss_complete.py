#!/usr/bin/env python3
"""
Test script for complete GPT-OSS-20B model
Verifies all components work together correctly
"""

import torch
import logging
import time
from src.moe.gpt_oss_model import GPTOSSModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_model_components():
    """Test individual model components"""
    logger.info("=" * 80)
    logger.info("Testing GPT-OSS-20B Complete Model")
    logger.info("=" * 80)

    # Initialize model
    logger.info("\n1. Initializing model...")
    start_time = time.time()
    model = GPTOSSModel("gpt-oss-20b/original")
    init_time = time.time() - start_time
    logger.info(f"✅ Model initialized in {init_time:.2f} seconds")

    # Model info
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"Total parameters: {total_params:.2f}B")

    # Move to GPU and set to eval mode
    model = model.cuda().eval()
    logger.info("Model moved to GPU and set to eval mode")

    # Test forward pass
    logger.info("\n2. Testing forward pass...")
    batch_size = 1
    seq_len = 10

    # Create test input
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
    logger.info(f"Input shape: {input_ids.shape}")

    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids)

    forward_time = time.time() - start_time

    # Check outputs
    if isinstance(outputs, dict):
        logits = outputs["logits"]
    else:
        logits = outputs

    logger.info(f"✅ Forward pass completed in {forward_time:.3f} seconds")
    logger.info(f"Output shape: {logits.shape}")
    logger.info(f"Output stats: mean={logits.mean():.4f}, std={logits.std():.4f}")
    logger.info(f"Output range: [{logits.min():.4f}, {logits.max():.4f}]")

    # Check for NaN/Inf
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()

    if has_nan or has_inf:
        logger.error(f"❌ Output contains NaN: {has_nan}, Inf: {has_inf}")
    else:
        logger.info("✅ Output is numerically stable (no NaN/Inf)")

    # Check output magnitude (should be reasonable for logits)
    if logits.std() > 100:
        logger.warning(f"⚠️ Output std is very high: {logits.std():.2f}")
    elif logits.std() < 0.1:
        logger.warning(f"⚠️ Output std is very low: {logits.std():.2f}")
    else:
        logger.info(f"✅ Output magnitude is reasonable (std={logits.std():.2f})")

    return model, logits


def test_generation():
    """Test text generation"""
    logger.info("\n3. Testing text generation...")

    # Initialize model
    model = GPTOSSModel("gpt-oss-20b/original").cuda().eval()

    # Test prompt
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).cuda()  # Simple test tokens
    logger.info(f"Input tokens: {input_ids.tolist()}")

    # Generate
    start_time = time.time()
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
    gen_time = time.time() - start_time

    logger.info(f"✅ Generated {generated.shape[1] - input_ids.shape[1]} tokens in {gen_time:.2f} seconds")
    logger.info(f"Generated shape: {generated.shape}")
    logger.info(f"Tokens/second: {(generated.shape[1] - input_ids.shape[1]) / gen_time:.2f}")


def test_memory_usage():
    """Test memory usage"""
    logger.info("\n4. Testing memory usage...")

    model = GPTOSSModel("gpt-oss-20b/original").cuda().eval()

    # Get memory stats
    mem_stats = model.get_memory_usage()

    logger.info("Memory Statistics:")
    for key, value in mem_stats.items():
        if "gb" in key.lower() or "parameters" in key:
            logger.info(f"  {key}: {value:.2f} GB")
        else:
            logger.info(f"  {key}: {value:.2f}")

    # Check VRAM usage
    vram_gb = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Total VRAM used: {vram_gb:.2f} GB")

    if vram_gb > 22:
        logger.warning(f"⚠️ High VRAM usage: {vram_gb:.2f} GB (target <22GB)")
    else:
        logger.info(f"✅ VRAM usage within limits: {vram_gb:.2f} GB")


def test_component_outputs():
    """Test outputs at each layer to debug magnitude issues"""
    logger.info("\n5. Testing component outputs...")

    model = GPTOSSModel("gpt-oss-20b/original").cuda().eval()

    # Create test input
    input_ids = torch.randint(0, 1000, (1, 10)).cuda()

    with torch.no_grad():
        # Test embeddings
        hidden = model.embedding(input_ids)
        logger.info(f"After embedding: mean={hidden.mean():.4f}, std={hidden.std():.4f}")

        # Test first few blocks
        for i in range(min(3, len(model.blocks))):
            hidden = model.blocks[i](hidden)
            logger.info(f"After block {i}: mean={hidden.mean():.4f}, std={hidden.std():.4f}")

            # Check for explosion
            if hidden.std() > 100:
                logger.error(f"❌ Activation explosion at block {i}")
                break

        # Final norm
        hidden = model.ln_f(hidden)
        logger.info(f"After final norm: mean={hidden.mean():.4f}, std={hidden.std():.4f}")

        # LM head
        logits = model.lm_head(hidden)
        logger.info(f"After lm_head: mean={logits.mean():.4f}, std={logits.std():.4f}")


if __name__ == "__main__":
    try:
        # Test components
        model, logits = test_model_components()

        # Test layer outputs
        test_component_outputs()

        # Test generation (optional, may be slow)
        # test_generation()

        # Test memory
        test_memory_usage()

        logger.info("\n" + "=" * 80)
        logger.info("✅ All tests completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}", exc_info=True)