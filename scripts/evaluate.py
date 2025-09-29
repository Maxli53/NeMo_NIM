#!/usr/bin/env python3
"""
Evaluation script for GPT-OSS-20B
Measures perplexity, generation quality, and comparison with base model
"""

import os
import argparse
import torch
import numpy as np
from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GPT-OSS-20B model")

    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str,
                       default="unsloth/gpt-oss-20b",
                       help="Base model for comparison")
    parser.add_argument("--dataset", type=str,
                       default="HuggingFaceH4/Multilingual-Thinking",
                       help="Evaluation dataset")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum samples to evaluate")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--output_file", type=str,
                       default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--compare_base", action="store_true",
                       help="Compare with base model")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed outputs")

    return parser.parse_args()

def calculate_perplexity(model, tokenizer, texts, max_length=2048):
    """Calculate perplexity on a dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating perplexity"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(model.device)

            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss

            total_loss += loss.item() * inputs.input_ids.size(1)
            total_tokens += inputs.input_ids.size(1)

    perplexity = np.exp(total_loss / total_tokens)
    return perplexity

def evaluate_generation_quality(model, tokenizer, prompts, args):
    """Evaluate generation quality with various metrics"""
    FastLanguageModel.for_inference(model)

    results = []

    for prompt in tqdm(prompts, desc="Evaluating generation"):
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip()

        # Calculate metrics
        response_length = len(tokenizer.encode(response))

        # Check for repetitions
        words = response.split()
        unique_ratio = len(set(words)) / len(words) if words else 0

        result = {
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "length": response_length,
            "unique_word_ratio": unique_ratio
        }

        results.append(result)

        if args.verbose:
            print(f"\nPrompt: {result['prompt']}")
            print(f"Response: {result['response']}")
            print(f"Metrics: Length={response_length}, Unique={unique_ratio:.2f}")

    return results

def compare_models(model_path, base_model_path, dataset, args):
    """Compare fine-tuned model with base model"""
    print("\nComparing models...")

    # Load both models
    print("Loading fine-tuned model...")
    finetuned_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True
    )

    print("Loading base model...")
    base_model, _ = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True
    )

    # Prepare test prompts
    test_prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
        "What are the benefits of meditation?",
        "How does photosynthesis work?"
    ]

    comparison = []

    FastLanguageModel.for_inference(finetuned_model)
    FastLanguageModel.for_inference(base_model)

    for prompt in test_prompts:
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        # Generate with fine-tuned
        with torch.no_grad():
            ft_outputs = finetuned_model.generate(
                **inputs, max_new_tokens=50, temperature=1.0
            )
        ft_response = tokenizer.decode(ft_outputs[0], skip_special_tokens=True)

        # Generate with base
        with torch.no_grad():
            base_outputs = base_model.generate(
                **inputs, max_new_tokens=50, temperature=1.0
            )
        base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)

        comparison.append({
            "prompt": prompt,
            "finetuned": ft_response[len(prompt):].strip()[:100],
            "base": base_response[len(prompt):].strip()[:100]
        })

    return comparison

def run_evaluation(args):
    """Run complete evaluation suite"""
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if os.path.exists(args.dataset):
        # Local dataset
        with open(args.dataset, 'r') as f:
            dataset = json.load(f)
    else:
        # HuggingFace dataset
        dataset = load_dataset(args.dataset, split="train")

    # Limit samples
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples] if isinstance(dataset, list) else dataset.select(range(args.max_samples))

    # Prepare texts for perplexity
    if isinstance(dataset, list):
        texts = [item.get("text", str(item)) for item in dataset]
    else:
        texts = dataset["text"] if "text" in dataset.features else [str(item) for item in dataset]

    texts = texts[:args.max_samples]

    # Calculate perplexity
    print("\n1. Calculating perplexity...")
    perplexity = calculate_perplexity(model, tokenizer, texts[:50])  # Use subset for speed
    print(f"   Perplexity: {perplexity:.2f}")

    # Evaluate generation quality
    print("\n2. Evaluating generation quality...")
    prompts = texts[:20]  # Use fewer for generation
    generation_results = evaluate_generation_quality(model, tokenizer, prompts, args)

    avg_length = np.mean([r["length"] for r in generation_results])
    avg_unique = np.mean([r["unique_word_ratio"] for r in generation_results])
    print(f"   Avg response length: {avg_length:.1f} tokens")
    print(f"   Avg unique word ratio: {avg_unique:.2%}")

    # Compare with base model if requested
    comparison_results = None
    if args.compare_base:
        comparison_results = compare_models(
            args.model_path, args.base_model, dataset, args
        )

    # Memory usage
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n3. Peak VRAM usage: {peak_memory:.2f} GB")

    # Compile results
    results = {
        "model": args.model_path,
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "perplexity": float(perplexity),
            "avg_response_length": float(avg_length),
            "avg_unique_ratio": float(avg_unique),
            "peak_memory_gb": float(peak_memory) if torch.cuda.is_available() else None,
            "samples_evaluated": len(texts)
        },
        "generation_samples": generation_results[:5],  # Save first 5
        "comparison": comparison_results
    }

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {args.output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Perplexity: {perplexity:.2f} (lower is better)")
    print(f"Response Quality: {avg_unique:.1%} unique words")

    if perplexity < 10:
        print("✅ Excellent perplexity!")
    elif perplexity < 20:
        print("✅ Good perplexity")
    else:
        print("⚠️ High perplexity - may need more training")

    return results

def main():
    args = parse_args()
    results = run_evaluation(args)

if __name__ == "__main__":
    main()