#!/usr/bin/env python3
"""
Unsloth GPT-OSS-20B Training Script V2 - With Fixes
- Proper GPU selection before imports
- Fixed LoftQ initialization
- Organized model saving to models/ folder
"""

import argparse
import os
from datetime import datetime

# Parse arguments FIRST, before any torch/CUDA imports
parser = argparse.ArgumentParser(description='Unsloth GPT-OSS-20B Training V2')
parser.add_argument('--profile', type=str, default='standard',
                   choices=['quick_test', 'standard', 'full', 'max_quality', 'conservative'],
                   help='Training profile to use')
parser.add_argument('--r', type=int, default=None,
                   help='LoRA rank (default: 8 for conservative, 16 for quality)')
parser.add_argument('--batch_size', type=int, default=None,
                   help='Batch size per device (default: based on profile)')
parser.add_argument('--learning_rate', type=float, default=2e-4,
                   help='Learning rate (default: 2e-4)')
parser.add_argument('--scheduler', type=str, default='cosine',
                   choices=['linear', 'cosine', 'cosine_with_restarts', 'constant'],
                   help='Learning rate scheduler (default: cosine)')
parser.add_argument('--max_steps', type=int, default=None,
                   help='Override max training steps')
parser.add_argument('--dataset_size', type=int, default=None,
                   help='Override dataset size')
parser.add_argument('--gpu', type=int, default=1,
                   help='GPU to use (default: 1)')
parser.add_argument('--model_name', type=str, default=None,
                   help='Custom name for saved model (default: auto-generated)')

args = parser.parse_args()

# SET GPU BEFORE ANY IMPORTS!
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
print(f"Using GPU {args.gpu}")

# NOW import torch/unsloth after GPU is set
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only
import torch

# Configuration profiles
PROFILES = {
    "quick_test": {
        "max_steps": 30,
        "dataset_size": 100,
        "save_steps": 15,
        "logging_steps": 1,
        "description": "Quick test run (30 steps, 100 samples)"
    },
    "standard": {
        "max_steps": 200,
        "dataset_size": 1000,
        "save_steps": 50,
        "logging_steps": 10,
        "description": "Standard training (200 steps, 1000 samples)"
    },
    "full": {
        "max_steps": 500,
        "dataset_size": 5000,
        "save_steps": 100,
        "logging_steps": 10,
        "description": "Full training (500 steps, 5000 samples)"
    },
    "max_quality": {
        "max_steps": 500,
        "dataset_size": 10000,
        "save_steps": 50,
        "logging_steps": 10,
        "r": 16,
        "batch_size": 6,
        "description": "Maximum quality (500 steps, 10k samples, r=16)"
    },
    "conservative": {
        "max_steps": 100,
        "dataset_size": 1000,
        "save_steps": 25,
        "logging_steps": 5,
        "r": 8,
        "batch_size": 2,
        "description": "Conservative/safe settings (r=8, batch=2)"
    }
}

def get_model_name(profile, scheduler, r):
    """
    Generate consistent model name using simplified convention.
    Technical details (rank, scheduler) are stored in training_info.txt
    """
    if args.model_name:
        return args.model_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # Simplified naming - details in training_info.txt
    return f"gpt-oss-20b_{profile}_{timestamp}"

def main():
    # Load profile configuration
    profile = PROFILES[args.profile]
    print("=" * 60)
    print(f"UNSLOTH TRAINING V2 - Profile: {args.profile}")
    print(f"Description: {profile['description']}")
    print("=" * 60)

    # Get configuration values (command-line args override profile)
    r = args.r or profile.get('r', 8 if args.profile == 'conservative' else 16)
    batch_size = args.batch_size or profile.get('batch_size', 2 if r == 8 else 4)
    max_steps = args.max_steps or profile['max_steps']
    dataset_size = args.dataset_size or profile['dataset_size']
    save_steps = profile.get('save_steps', 50)
    logging_steps = profile.get('logging_steps', 10)

    # Calculate gradient accumulation for effective batch size ~16
    grad_accum = max(1, 16 // batch_size)

    # Generate model name
    model_name = get_model_name(args.profile, args.scheduler, r)
    output_dir = f"models/{model_name}"

    print(f"Configuration:")
    print(f"  LoRA rank: {r}")
    print(f"  Batch size: {batch_size} x {grad_accum} = {batch_size * grad_accum} effective")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Scheduler: {args.scheduler}")
    print(f"  Max steps: {max_steps}")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Output: {output_dir}")
    print(f"  GPU: {args.gpu}")
    print()

    # ============================================================
    # STEP 1: Load Pre-Quantized Model
    # ============================================================
    max_seq_length = 2048 if r >= 16 else 1024

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # ============================================================
    # STEP 2: Add LoRA Adapters (WITH FIXES)
    # ============================================================
    lora_alpha = r * 2  # 2x rank

    # Build configuration
    peft_config = {
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
    }

    # Add advanced features for higher ranks
    if r >= 16:
        peft_config["use_rslora"] = True
        # Note: LoftQ requires FP16 loading (40GB VRAM for 20B model)
        # We skip it but keep config for compatibility (will show warning)
        peft_config["loftq_config"] = {
            "loftq_bits": 4,
            "loftq_iter": 1,
        }
        print(f"Applying LoRA with r={r}, alpha={lora_alpha}")
        print("  Using RSLoRA for rank stability")
        print("  Note: LoftQ config present but inactive with 4-bit models")
    else:
        print(f"Applying LoRA with r={r}, alpha={lora_alpha}")

    model = FastLanguageModel.get_peft_model(model, **peft_config)

    # ============================================================
    # STEP 3: Load Dataset
    # ============================================================
    print(f"Loading dataset ({dataset_size} samples)...")
    dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking",
                           split=f"train[:{dataset_size}]")
    dataset = standardize_sharegpt(dataset)

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = []
        for convo in convos:
            text = tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
                reasoning_effort="medium",
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # ============================================================
    # STEP 4: Training Configuration
    # ============================================================
    warmup_steps = 20 if args.scheduler == "cosine" else 10

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            # Core settings
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,

            # Learning settings
            learning_rate=args.learning_rate,
            max_steps=max_steps,

            # Optimizer
            optim="adamw_8bit",
            weight_decay=0.01,

            # Scheduler (defaulting to cosine)
            lr_scheduler_type=args.scheduler,
            warmup_steps=warmup_steps,

            # Precision
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),

            # Speed optimization
            group_by_length=True,

            # Logging and saving
            logging_steps=logging_steps,
            output_dir=output_dir,  # Save to models/ folder
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,

            # Other
            seed=3407,
            max_seq_length=max_seq_length,
            packing=False,
        ),
    )

    # ============================================================
    # STEP 5: Train on Completions Only
    # ============================================================
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start|>user<|message|>",
        response_part="<|start|>assistant<|channel|>",
    )

    # ============================================================
    # STEP 6: Show Stats and Train
    # ============================================================
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)  # 0 because we set CUDA_VISIBLE_DEVICES
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"\nGPU: {gpu_stats.name}")
        print(f"Memory Allocated: {memory_allocated:.2f}GB")
        print(f"Max Memory: {gpu_stats.total_memory / 1024**3:.2f}GB")

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    # Train!
    trainer_stats = trainer.train()

    # ============================================================
    # STEP 7: Save Final Model
    # ============================================================
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training info
    info_path = f"{output_dir}/training_info.txt"
    with open(info_path, 'w') as f:
        f.write(f"Model: GPT-OSS-20B\n")
        f.write(f"Profile: {args.profile}\n")
        f.write(f"LoRA rank: {r}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Scheduler: {args.scheduler}\n")
        f.write(f"Max steps: {max_steps}\n")
        f.write(f"Dataset size: {dataset_size}\n")
        f.write(f"Final loss: {trainer_stats.training_loss:.4f}\n")
        f.write(f"GPU used: {args.gpu}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Final loss: {trainer_stats.training_loss:.4f}")
    print(f"Model saved to: {output_dir}/")

    # Quick test
    print("\nRunning quick test...")
    FastLanguageModel.for_inference(model)
    inputs = tokenizer("What is 2+2?", return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test response: {response}")

    # Create a symlink to latest model
    latest_link = "models/latest"
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(model_name, latest_link)
    print(f"\nCreated symlink: models/latest -> {model_name}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("Available profiles:")
        for name, config in PROFILES.items():
            print(f"  --profile {name}: {config['description']}")
        print("\nExample usage:")
        print("  python train_standard.py --profile quick_test")
        print("  python train_standard.py --profile standard --scheduler cosine")
        print("  python train_standard.py --profile max_quality --gpu 1")
        print("\nModels will be saved to: models/{profile}_{timestamp}/")
    else:
        main()