#!/usr/bin/env python3
"""
GPT-OSS-20B Training Script using Unsloth
Based on official notebook: https://github.com/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb

Memory Requirements:
- QLoRA (4-bit): 14GB VRAM (single RTX 3090)
- BF16 LoRA: 44GB VRAM (both RTX 3090s)
"""

import os
import argparse
import torch
from unsloth import FastLanguageModel, UnslothTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-OSS-20B with Unsloth")

    # Model arguments
    parser.add_argument("--model_name", type=str,
                       default="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
                       help="Model name or path")
    parser.add_argument("--max_seq_length", type=int, default=16384,
                       help="Maximum sequence length (min 16384 recommended)")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                       help="Load model in 4-bit (required for single GPU)")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank (8 from official notebook)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0,
                       help="LoRA dropout")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=100,
                       help="Maximum training steps (-1 for no limit)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=5,
                       help="Warmup steps")

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str,
                       default="HuggingFaceH4/Multilingual-Thinking",
                       help="Dataset name from HuggingFace")
    parser.add_argument("--dataset_split", type=str, default="train",
                       help="Dataset split to use")
    parser.add_argument("--max_samples", type=int, default=-1,
                       help="Maximum number of samples to use (-1 for all)")

    # Output arguments
    parser.add_argument("--output_dir", type=str,
                       default="./checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--final_output_dir", type=str,
                       default="./final_model",
                       help="Final model output directory")

    # GPT-OSS specific
    parser.add_argument("--reasoning_effort", type=str, default="medium",
                       choices=["low", "medium", "high"],
                       help="Reasoning effort level for GPT-OSS")

    # Other arguments
    parser.add_argument("--use_dual_gpu", action="store_true",
                       help="Use both RTX 3090s")
    parser.add_argument("--seed", type=int, default=3407,  # Official seed
                       help="Random seed")

    return parser.parse_args()

def setup_environment(args):
    """Setup CUDA and environment"""
    if args.use_dual_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        print("Using both RTX 3090s (48GB total)")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("Using single RTX 3090 (24GB)")

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Print GPU info
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")

def load_model(args):
    """Load model with Unsloth optimizations"""
    print(f"\nLoading model: {args.model_name}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"4-bit quantization: {args.load_in_4bit}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=args.load_in_4bit,
    )

    print("Model loaded successfully!")
    return model, tokenizer

def add_lora_adapters(model, args):
    """Add LoRA adapters to model"""
    print(f"\nAdding LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimization
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    print("LoRA adapters added!")
    return model

def prepare_dataset(args, tokenizer):
    """Load and prepare dataset"""
    print(f"\nLoading dataset: {args.dataset_name}")

    # Load dataset
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    # Limit samples if specified
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Dataset size: {len(dataset)} samples")

    # Standardize dataset format (from official docs)
    from unsloth.chat_templates import standardize_sharegpt
    dataset = standardize_sharegpt(dataset)

    # Format dataset for chat template (official method)
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
            reasoning_effort=args.reasoning_effort  # GPT-OSS specific
        ) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Print first example for verification
    if len(dataset) > 0:
        print("\nFirst training example:")
        print(dataset[0]['text'][:500] + "..." if len(dataset[0]['text']) > 500 else dataset[0]['text'])

    return dataset

def train_model(model, tokenizer, dataset, args):
    """Train the model"""
    print(f"\nStarting training...")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Max steps: {args.max_steps}")

    # Training arguments optimized for RTX 3090
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=True,  # Mixed precision
        optim="adamw_8bit",  # 8-bit optimizer for memory efficiency
        warmup_steps=args.warmup_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        seed=args.seed,
        report_to=["tensorboard"],
        load_best_model_at_end=True if args.max_steps > 50 else False,
    )

    # Create trainer
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=lambda data: {
            'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
            'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
            'labels': torch.stack([torch.tensor(f['labels']) for f in data])
        } if 'input_ids' in data[0] else None,
    )

    # Print memory usage before training
    if torch.cuda.is_available():
        print(f"\nMemory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Train
    train_result = trainer.train()

    # Print memory usage after training
    if torch.cuda.is_available():
        print(f"Peak memory during training: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    return trainer, train_result

def save_model(model, tokenizer, args):
    """Save the fine-tuned model"""
    print(f"\nSaving model to {args.final_output_dir}")

    # Save LoRA adapters
    model.save_pretrained(args.final_output_dir)
    tokenizer.save_pretrained(args.final_output_dir)

    # Save training info
    info = {
        "model_name": args.model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_seq_length": args.max_seq_length,
        "training_steps": args.max_steps,
        "timestamp": datetime.now().isoformat(),
    }

    with open(f"{args.final_output_dir}/training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("Model saved successfully!")

    # Option to export to GGUF
    response = input("\nExport to GGUF format? (y/n): ")
    if response.lower() == 'y':
        export_to_gguf(model, tokenizer, args)

def export_to_gguf(model, tokenizer, args):
    """Export model to GGUF format"""
    print("\nExporting to GGUF format...")

    gguf_dir = f"{args.final_output_dir}_gguf"

    # Q4_K_M is recommended (best quality/size ratio)
    model.save_pretrained_gguf(
        gguf_dir,
        tokenizer,
        quantization_method="q4_k_m"
    )

    print(f"GGUF model saved to {gguf_dir}")
    print("You can now use this with llama.cpp!")

def main():
    args = parse_args()

    print("=" * 60)
    print("GPT-OSS-20B Training with Unsloth")
    print("=" * 60)

    # Setup environment
    setup_environment(args)

    # Load model
    model, tokenizer = load_model(args)

    # Add LoRA adapters
    model = add_lora_adapters(model, args)

    # Prepare dataset
    dataset = prepare_dataset(args, tokenizer)

    # Train
    trainer, train_result = train_model(model, tokenizer, dataset, args)

    # Save model
    save_model(model, tokenizer, args)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final loss: {train_result.training_loss:.4f}")
    print(f"Total training time: {train_result.metrics['train_runtime']:.2f} seconds")
    print("=" * 60)

if __name__ == "__main__":
    main()