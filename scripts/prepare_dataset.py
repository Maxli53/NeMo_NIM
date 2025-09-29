#!/usr/bin/env python3
"""
Dataset preparation script for GPT-OSS-20B
Converts various formats to Unsloth-compatible format
"""

import os
import json
import argparse
from datasets import Dataset, load_dataset
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare datasets for Unsloth training")

    parser.add_argument("--input_file", type=str, required=True,
                       help="Input file (JSON, JSONL, CSV, or HF dataset)")
    parser.add_argument("--output_dir", type=str, default="./data/processed",
                       help="Output directory")
    parser.add_argument("--format", type=str,
                       choices=["alpaca", "sharegpt", "openai", "raw"],
                       default="sharegpt",
                       help="Input format")
    parser.add_argument("--train_split", type=float, default=0.9,
                       help="Training split ratio")
    parser.add_argument("--max_samples", type=int, default=-1,
                       help="Maximum samples to process")
    parser.add_argument("--validate", action="store_true",
                       help="Validate dataset quality")

    return parser.parse_args()

def convert_alpaca_to_sharegpt(data):
    """Convert Alpaca format to ShareGPT format"""
    messages = []
    for item in data:
        conversation = []

        # System prompt if exists
        if item.get("input", ""):
            conversation.append({
                "role": "system",
                "content": item["input"]
            })

        # User instruction
        conversation.append({
            "role": "user",
            "content": item["instruction"]
        })

        # Assistant response
        conversation.append({
            "role": "assistant",
            "content": item["output"]
        })

        messages.append({"messages": conversation})

    return messages

def convert_openai_to_sharegpt(data):
    """Convert OpenAI format to ShareGPT format"""
    return [{"messages": item["messages"]} for item in data]

def load_raw_data(file_path, format_type):
    """Load data from various formats"""
    ext = Path(file_path).suffix.lower()

    if ext == ".json":
        with open(file_path, 'r') as f:
            data = json.load(f)
    elif ext == ".jsonl":
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
    else:
        # Try loading as HuggingFace dataset
        dataset = load_dataset(file_path)
        data = dataset['train'] if 'train' in dataset else dataset

    return data

def validate_dataset(data):
    """Validate dataset quality and structure"""
    issues = []
    stats = {
        "total_samples": len(data),
        "avg_turns": 0,
        "max_length": 0,
        "min_length": float('inf'),
        "empty_responses": 0
    }

    total_turns = 0

    for idx, item in enumerate(data):
        if "messages" not in item:
            issues.append(f"Sample {idx}: Missing 'messages' field")
            continue

        messages = item["messages"]
        total_turns += len(messages)

        for msg in messages:
            if "role" not in msg or "content" not in msg:
                issues.append(f"Sample {idx}: Invalid message format")

            content_length = len(msg.get("content", ""))
            stats["max_length"] = max(stats["max_length"], content_length)
            stats["min_length"] = min(stats["min_length"], content_length)

            if content_length == 0:
                stats["empty_responses"] += 1

    stats["avg_turns"] = total_turns / len(data) if data else 0

    return stats, issues

def add_reasoning_effort(data, distribution=None):
    """Add reasoning effort to messages (GPT-OSS specific)"""
    import random

    if distribution is None:
        distribution = {"low": 0.25, "medium": 0.50, "high": 0.25}

    efforts = ["low", "medium", "high"]
    probs = [distribution[e] for e in efforts]

    for item in data:
        # Add reasoning effort to system messages
        effort = random.choices(efforts, weights=probs)[0]

        for msg in item["messages"]:
            if msg["role"] == "system":
                if "Reasoning:" not in msg["content"]:
                    msg["content"] = f"Reasoning: {effort}\n{msg['content']}"

    return data

def main():
    args = parse_args()

    print("=" * 60)
    print("Dataset Preparation for Unsloth")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from: {args.input_file}")
    raw_data = load_raw_data(args.input_file, args.format)

    # Limit samples if specified
    if args.max_samples > 0:
        raw_data = raw_data[:args.max_samples]

    print(f"Loaded {len(raw_data)} samples")

    # Convert to ShareGPT format
    print(f"\nConverting from {args.format} to ShareGPT format...")

    if args.format == "alpaca":
        data = convert_alpaca_to_sharegpt(raw_data)
    elif args.format == "openai":
        data = convert_openai_to_sharegpt(raw_data)
    elif args.format == "sharegpt":
        data = raw_data  # Already in correct format
    else:
        # Raw format - user must structure it properly
        data = raw_data

    # Add reasoning effort for GPT-OSS
    print("\nAdding reasoning effort levels...")
    data = add_reasoning_effort(data)

    # Validate if requested
    if args.validate:
        print("\nValidating dataset...")
        stats, issues = validate_dataset(data)

        print(f"\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        if issues:
            print(f"\n⚠️ Found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10
                print(f"  - {issue}")

    # Split dataset
    split_idx = int(len(data) * args.train_split)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    print(f"\nSplit: {len(train_data)} train, {len(eval_data)} eval")

    # Save processed datasets
    os.makedirs(args.output_dir, exist_ok=True)

    # Save as JSON for easy loading
    train_path = os.path.join(args.output_dir, "train.json")
    eval_path = os.path.join(args.output_dir, "eval.json")

    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, indent=2)

    print(f"\nSaved to:")
    print(f"  Training: {train_path}")
    print(f"  Evaluation: {eval_path}")

    # Create HuggingFace dataset
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    # Save as HF dataset
    hf_path = os.path.join(args.output_dir, "hf_dataset")
    train_dataset.save_to_disk(os.path.join(hf_path, "train"))
    eval_dataset.save_to_disk(os.path.join(hf_path, "eval"))

    print(f"  HF Dataset: {hf_path}")

    print("\n✅ Dataset preparation complete!")
    print("\nTo use in training:")
    print(f"  python scripts/train_simple.py \\")
    print(f"    --dataset_name {hf_path}/train \\")
    print(f"    --max_steps 100")

if __name__ == "__main__":
    main()