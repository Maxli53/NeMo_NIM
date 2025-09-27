#!/usr/bin/env python3
"""
Convert GPT-OSS-20B from HuggingFace to NeMo format
Based on official NeMo tutorial
"""
import os
import nemo_run as run
from nemo.collections import llm

# Set cache directory
NEMO_MODELS_CACHE = "/workspace/checkpoints"
os.environ['NEMO_MODELS_CACHE'] = NEMO_MODELS_CACHE

print("=" * 70)
print("GPT-OSS-20B HuggingFace → NeMo Conversion")
print("=" * 70)
print(f"Source: /models/gpt-oss-20b (HuggingFace)")
print(f"Target: {NEMO_MODELS_CACHE}/gpt-oss-20b (NeMo)")
print("=" * 70)

def configure_checkpoint_conversion():
    return run.Partial(
        llm.import_ckpt,
        model=run.Config(llm.GPTOSSModel, run.Config(llm.GPTOSSConfig20B)),
        source="hf:///models/gpt-oss-20b",
        overwrite=False,
    )

print("\nStarting conversion (this may take 10-15 minutes)...\n")

# Run the conversion locally
run.run(configure_checkpoint_conversion(), executor=run.LocalExecutor())

print("\n" + "=" * 70)
print("Conversion Complete!")
print(f"NeMo checkpoint saved to: {NEMO_MODELS_CACHE}/gpt-oss-20b")
print("=" * 70)