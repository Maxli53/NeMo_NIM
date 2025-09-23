#!/usr/bin/env python3
"""
Safe test script for GPT-OSS-20B generation
Tests memory-safe generation with proper error handling
"""

import torch
import logging
import gc
from src.moe.gpt_oss_model import GPTOSSModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_safe_generation():
    """Test generation with memory safety measures"""

    logger.info("=" * 60)
    logger.info("Testing Safe Generation for GPT-OSS-20B")
    logger.info("=" * 60)

    # Clear any existing memory
    torch.cuda.empty_cache()
    gc.collect()

    initial_memory = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Initial GPU memory: {initial_memory:.2f} GB")

    try:
        # Initialize model
        logger.info("\n1. Loading model...")
        model = GPTOSSModel("gpt-oss-20b/original").cuda().eval()

        model_memory = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Model loaded, memory usage: {model_memory:.2f} GB")

        # Test with small input first
        logger.info("\n2. Testing with small input (5 tokens)...")
        input_ids = torch.tensor([[100, 200, 300, 400, 500]]).cuda()

        # Test forward pass first
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            logger.info(f"Forward pass successful, output shape: {logits.shape}")
            del outputs, logits
            torch.cuda.empty_cache()

        # Test generation with small number of tokens
        logger.info("\n3. Generating 5 tokens...")
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=5,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                max_context_length=128  # Small context for safety
            )

        logger.info(f"Generated successfully: {generated.shape}")
        logger.info(f"Generated tokens: {generated[0].tolist()}")

        gen_memory = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Memory after generation: {gen_memory:.2f} GB")

        # Clear memory
        del generated
        torch.cuda.empty_cache()

        # Test with slightly longer generation
        logger.info("\n4. Testing longer generation (10 tokens)...")
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                max_context_length=256
            )

        logger.info(f"Generated {generated.shape[1] - input_ids.shape[1]} new tokens")

        final_memory = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Final memory usage: {final_memory:.2f} GB")

        # Check memory didn't explode
        if final_memory < 22:
            logger.info("\n✅ SUCCESS: Generation completed without memory issues!")
            logger.info(f"Memory stayed under control: {final_memory:.2f}GB < 22GB")
        else:
            logger.warning(f"⚠️ High memory usage: {final_memory:.2f}GB")

        return True

    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

        # Try to clean up
        torch.cuda.empty_cache()
        gc.collect()

        return False

    finally:
        # Final cleanup
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        gc.collect()

        final_cleanup_memory = torch.cuda.memory_allocated() / 1e9
        logger.info(f"\nMemory after cleanup: {final_cleanup_memory:.2f} GB")


def test_stress_generation():
    """Stress test with longer sequences (optional)"""

    logger.info("\n" + "=" * 60)
    logger.info("Stress Testing Generation (Optional)")
    logger.info("=" * 60)

    try:
        model = GPTOSSModel("gpt-oss-20b/original").cuda().eval()

        # Start with moderate input
        input_ids = torch.randint(100, 1000, (1, 20)).cuda()

        logger.info(f"Starting with {input_ids.shape[1]} tokens")
        logger.info("Generating 20 tokens with sliding window...")

        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.8,
                max_context_length=512  # Sliding window
            )

        logger.info(f"✅ Generated {generated.shape[1] - input_ids.shape[1]} tokens successfully")
        logger.info(f"Total sequence length: {generated.shape[1]} tokens")

        memory_used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Memory usage: {memory_used:.2f} GB")

        return True

    except Exception as e:
        logger.error(f"Stress test failed (expected): {e}")
        return False

    finally:
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    # Run safe generation test
    success = test_safe_generation()

    if success:
        logger.info("\n" + "=" * 60)
        logger.info("SAFE GENERATION TEST PASSED!")
        logger.info("=" * 60)

        # Optional: run stress test
        # Uncomment to test longer sequences
        # test_stress_generation()
    else:
        logger.error("\nGeneration test failed - check memory management")