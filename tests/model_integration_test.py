#!/usr/bin/env python3
"""
Working GPT-OSS test with correct dtype (bfloat16)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_memory():
    if torch.cuda.is_available():
        logger.info(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
    logger.info(f"RAM: {psutil.Process().memory_info().rss/1e9:.2f} GB used")

logger.info("Testing GPT-OSS with correct dtype")
logger.info("=" * 60)

# Check CUDA
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name()}")
    logger.info(f"CUDA version: {torch.version.cuda}")

print_memory()

try:
    # Path to the HuggingFace format model
    model_path = Path("C:/Users/maxli/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee")

    logger.info(f"\nModel path: {model_path}")

    # Load tokenizer
    logger.info("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("   ✓ Tokenizer loaded!")

    # Load model with bfloat16 (required for GPT-OSS)
    logger.info("\n2. Loading model with bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 instead of float16
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        offload_folder="offload",
        max_memory={0: "20GB", "cpu": "30GB"}
    )
    logger.info("   ✓ Model loaded successfully!")

    print_memory()

    # Model info
    logger.info("\n3. Model Information:")
    logger.info(f"   - Model type: GPT-OSS 20B MoE")
    logger.info(f"   - 21B total parameters")
    logger.info(f"   - 3.6B active parameters per token")
    logger.info(f"   - 32 experts, 4 active per token")
    logger.info(f"   - Using bfloat16 precision")

    # Test generation
    logger.info("\n4. Testing text generation...")

    test_prompts = [
        "Explain quantum computing in simple terms:",
        "The future of artificial intelligence is",
        "How do mixture of experts models work?",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\n   Test {i}/{len(test_prompts)}")
        logger.info(f"   Prompt: '{prompt}'")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.95
            )

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"   Response: {response[:200]}...")

    print_memory()

    logger.info("\n" + "=" * 60)
    logger.info("SUCCESS: GPT-OSS is working with transformers!")
    logger.info("=" * 60)

    logger.info("\nNext steps:")
    logger.info("1. Integrate with multi-agent framework")
    logger.info("2. Add PDF loading for knowledge bases")
    logger.info("3. Implement LoRA fine-tuning")
    logger.info("4. Create web interface")

except Exception as e:
    logger.error(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()