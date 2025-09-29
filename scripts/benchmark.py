#!/usr/bin/env python3
"""
Benchmark script for GPT-OSS-20B
Tests memory usage, speed, and quality
"""

import os
import time
import torch
import argparse
from unsloth import FastLanguageModel
import json

def benchmark_model(model_path="unsloth/gpt-oss-20b", max_seq_length=2048):
    """Run comprehensive benchmarks"""

    print("=" * 60)
    print(f"Benchmarking: {model_path}")
    print("=" * 60)

    # Load model
    start_load = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    load_time = time.time() - start_load

    print(f"✓ Model load time: {load_time:.2f}s")

    # Memory benchmark
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"✓ VRAM allocated: {allocated:.2f} GB")
        print(f"✓ VRAM reserved: {reserved:.2f} GB")

    # Speed benchmark
    test_prompts = [
        "What is 2+2?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about AI.",
    ]

    total_tokens = 0
    total_time = 0

    for prompt in test_prompts:
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, temperature=1.0)
        gen_time = time.time() - start

        tokens = len(outputs[0]) - len(inputs.input_ids[0])
        total_tokens += tokens
        total_time += gen_time

    avg_speed = total_tokens / total_time if total_time > 0 else 0
    print(f"✓ Average speed: {avg_speed:.1f} tokens/s")

    # Peak memory
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"✓ Peak VRAM: {peak:.2f} GB")

    results = {
        "model": model_path,
        "load_time_s": load_time,
        "vram_allocated_gb": allocated if torch.cuda.is_available() else 0,
        "peak_vram_gb": peak if torch.cuda.is_available() else 0,
        "tokens_per_second": avg_speed,
        "max_seq_length": max_seq_length,
    }

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="unsloth/gpt-oss-20b")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--save_results", default="benchmark_results.json")
    args = parser.parse_args()

    results = benchmark_model(args.model_path, args.max_seq_length)

    with open(args.save_results, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.save_results}")