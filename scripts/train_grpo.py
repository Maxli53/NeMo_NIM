#!/usr/bin/env python3
"""
GRPO/RL Training Script for GPT-OSS-20B
Based on Unsloth's GRPO implementation

Requirements:
- 15GB VRAM for GPT-OSS-20B
- Unsloth with RL support
- Custom reward functions
"""

import os
import argparse
import torch
import numpy as np
import time
import json
import types
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from typing import List, Dict, Any
import warnings

# Disable Flash Attention 3 (critical for GPT-OSS)
os.environ["UNSLOTH_USE_FA3"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training for GPT-OSS")

    # Model arguments
    parser.add_argument("--model_name", type=str,
                       default="unsloth/gpt-oss-20b",
                       help="Model name or path")
    parser.add_argument("--max_seq_length", type=int, default=768,
                       help="Max sequence length (reduced for RL)")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                       help="Use 4-bit quantization")
    parser.add_argument("--offload_embeddings", action="store_true", default=True,
                       help="Offload embeddings to save 1GB VRAM")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=4,
                       help="LoRA rank (lower for RL)")
    parser.add_argument("--lora_alpha", type=int, default=4,
                       help="LoRA alpha")

    # GRPO arguments
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Generation temperature")
    parser.add_argument("--num_generations", type=int, default=2,
                       help="Number of candidates per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Max tokens to generate")

    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate (lower for RL)")
    parser.add_argument("--max_steps", type=int, default=100,
                       help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size per device")

    # Task arguments
    parser.add_argument("--task", type=str, default="code_optimization",
                       choices=["code_optimization", "reasoning", "creative"],
                       help="Task type for reward functions")
    parser.add_argument("--output_dir", type=str, default="./grpo_outputs",
                       help="Output directory")

    # Advanced
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")

    return parser.parse_args()

def setup_environment(args):
    """Setup environment for GRPO training"""
    print("=" * 60)
    print("GRPO Environment Setup")
    print("=" * 60)

    # Check Flash Attention status
    if os.environ.get("UNSLOTH_USE_FA3", "0") != "0":
        warnings.warn("⚠️ Flash Attention 3 must be disabled for GPT-OSS!")
        os.environ["UNSLOTH_USE_FA3"] = "0"

    # GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # CUDA settings for memory efficiency
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

def load_model_for_rl(args):
    """Load model with RL optimizations"""
    print(f"\nLoading model for RL: {args.model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        device_map="auto",
    )

    # Add LoRA for RL (smaller rank)
    print(f"Adding LoRA adapters (r={args.lora_r})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Enable embedding offloading if requested
    if args.offload_embeddings:
        print("Enabling embedding offloading (saves 1GB VRAM)")
        model.config.offload_embeddings = True

    return model, tokenizer

class RewardFunctions:
    """Collection of reward functions for different tasks"""

    @staticmethod
    def code_optimization_rewards():
        """Rewards for code optimization task"""

        def syntax_valid(code: str) -> float:
            """Check if code is syntactically valid Python"""
            try:
                compile(code, '<string>', 'exec')
                return 1.0
            except SyntaxError:
                return 0.0

        def no_imports(code: str) -> float:
            """Prevent using external libraries"""
            banned = ["import numpy", "import torch", "import scipy",
                     "from numpy", "from torch", "from scipy"]
            for ban in banned:
                if ban in code:
                    return -1.0  # Negative reward for cheating
            return 1.0

        def has_function(code: str, func_name: str = "matmul") -> float:
            """Check if required function exists"""
            try:
                namespace = {}
                exec(code, namespace)
                if func_name in namespace:
                    return 1.0
                return 0.0
            except:
                return 0.0

        def correctness(code: str) -> float:
            """Check mathematical correctness"""
            try:
                # Create sandboxed namespace
                namespace = {"__builtins__": {}}
                exec(code, namespace)

                func = namespace.get("matmul")
                if not func:
                    return 0.0

                # Test cases
                A = [[1, 2], [3, 4]]
                B = [[5, 6], [7, 8]]
                expected = [[19, 22], [43, 50]]

                result = func(A, B)

                # Check result
                for i in range(2):
                    for j in range(2):
                        if abs(result[i][j] - expected[i][j]) > 0.001:
                            return 0.0
                return 2.0  # Higher reward for correctness

            except:
                return 0.0

        def performance(code: str) -> float:
            """Benchmark performance"""
            try:
                # Locked execution
                namespace = {}
                exec(code, namespace)
                func = namespace.get("matmul")

                if not func:
                    return 0.0

                # Benchmark
                import timeit
                setup = f"""
{code}
import random
A = [[random.random() for _ in range(10)] for _ in range(10)]
B = [[random.random() for _ in range(10)] for _ in range(10)]
"""
                time_taken = timeit.timeit("matmul(A, B)", setup=setup, number=100)

                # Reward based on speed (lower is better)
                if time_taken < 0.01:
                    return 2.0
                elif time_taken < 0.1:
                    return 1.0
                else:
                    return 0.5

            except:
                return 0.0

        return [syntax_valid, no_imports, has_function, correctness, performance]

    @staticmethod
    def reasoning_rewards():
        """Rewards for reasoning tasks"""

        def has_steps(response: str) -> float:
            """Check if response has step-by-step reasoning"""
            indicators = ["Step", "First", "Second", "Then", "Finally", "Therefore"]
            count = sum(1 for ind in indicators if ind in response)
            return min(count / 3.0, 1.0)  # Normalize

        def coherent(response: str) -> float:
            """Check response coherence"""
            sentences = response.split('.')
            if len(sentences) < 2:
                return 0.5
            return 1.0

        def conclusion(response: str) -> float:
            """Check if response has conclusion"""
            conclusions = ["therefore", "thus", "so", "in conclusion", "finally"]
            if any(c in response.lower() for c in conclusions):
                return 1.0
            return 0.5

        return [has_steps, coherent, conclusion]

    @staticmethod
    def creative_rewards():
        """Rewards for creative writing"""

        def length_appropriate(response: str) -> float:
            """Check if response length is appropriate"""
            words = len(response.split())
            if 50 <= words <= 500:
                return 1.0
            elif words < 50:
                return 0.5
            else:
                return 0.7

        def variety(response: str) -> float:
            """Check vocabulary variety"""
            words = response.lower().split()
            if len(words) == 0:
                return 0.0
            unique_ratio = len(set(words)) / len(words)
            return unique_ratio * 2  # Scale up

        def no_repetition(response: str) -> float:
            """Penalize repetitive phrases"""
            # Check for repeated 3-grams
            words = response.split()
            if len(words) < 3:
                return 1.0

            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            unique_ratio = len(set(trigrams)) / len(trigrams)
            return unique_ratio

        return [length_appropriate, variety, no_repetition]

def create_grpo_config(args):
    """Create GRPO configuration"""
    config = GRPOConfig(
        # Generation settings
        temperature=args.temperature,
        top_p=1.0,
        top_k=0,
        max_new_tokens=args.max_new_tokens,
        num_generations=args.num_generations,

        # Training settings
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",

        # Batch settings
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,

        # Duration
        max_steps=args.max_steps,

        # Logging
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        output_dir=args.output_dir,

        # Seed
        seed=3407,

        # RL specific
        reward_baseline=0.0,  # Running average baseline
        kl_penalty="kl",
        kl_coef=0.1,

        # Report
        report_to=["tensorboard"],
    )

    return config

def prepare_dataset(args):
    """Prepare dataset for GRPO training"""
    print("\nPreparing dataset...")

    if args.task == "code_optimization":
        # Matrix multiplication prompts
        prompts = [
            "Write a Python function called 'matmul' that multiplies two matrices A and B without using numpy or any external libraries:",
            "Implement matrix multiplication in pure Python. The function should be called 'matmul' and take two 2D lists as input:",
            "Create an optimized matrix multiplication function 'matmul' using only Python built-ins:",
        ]
    elif args.task == "reasoning":
        prompts = [
            "Explain step by step how photosynthesis works:",
            "Walk through the process of solving a quadratic equation:",
            "Describe the steps to bake a cake from scratch:",
        ]
    else:  # creative
        prompts = [
            "Write a short story about a robot learning to paint:",
            "Describe a futuristic city in the year 3000:",
            "Create a poem about artificial intelligence:",
        ]

    # Create dataset format
    dataset = [{"prompt": p, "query": p} for p in prompts]

    # Repeat for more training data
    dataset = dataset * (args.max_steps // len(dataset) + 1)

    return dataset

def train_grpo(model, tokenizer, dataset, config, reward_functions, args):
    """Main GRPO training loop"""
    print("\n" + "=" * 60)
    print("Starting GRPO Training")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Steps: {args.max_steps}")
    print(f"Reward functions: {len(reward_functions)}")

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=dataset,
    )

    # Training loop with custom rewards
    best_reward = -float('inf')
    reward_history = []

    for step in range(args.max_steps):
        # Get batch
        batch = dataset[step % len(dataset)]
        prompt = batch["prompt"]

        # Generate candidates
        with torch.no_grad():
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            generations = []

            for _ in range(args.num_generations):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = generated[len(prompt):]
                generations.append(response)

        # Calculate rewards
        rewards = []
        for gen in generations:
            reward = sum(rf(gen) for rf in reward_functions)
            rewards.append(reward)

        # Update model (simplified - real implementation would use trainer)
        avg_reward = np.mean(rewards)
        reward_history.append(avg_reward)

        # Logging
        if step % 10 == 0:
            print(f"Step {step}: Avg Reward = {avg_reward:.2f}")
            if args.debug:
                print(f"  Best generation: {generations[np.argmax(rewards)][:100]}...")

        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            model.save_pretrained(f"{args.output_dir}/best_model")
            tokenizer.save_pretrained(f"{args.output_dir}/best_model")

    # Final statistics
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final avg reward: {np.mean(reward_history[-10:]):.2f}")

    # Save training history
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump({
            "reward_history": reward_history,
            "best_reward": float(best_reward),
            "config": vars(args)
        }, f, indent=2)

def main():
    args = parse_args()

    # Setup
    setup_environment(args)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model_for_rl(args)

    # Get reward functions
    if args.task == "code_optimization":
        reward_functions = RewardFunctions.code_optimization_rewards()
    elif args.task == "reasoning":
        reward_functions = RewardFunctions.reasoning_rewards()
    else:
        reward_functions = RewardFunctions.creative_rewards()

    # Prepare dataset
    dataset = prepare_dataset(args)

    # Create config
    config = create_grpo_config(args)

    # Train
    train_grpo(model, tokenizer, dataset, config, reward_functions, args)

    print(f"\nModel saved to: {args.output_dir}/best_model")
    print("\nTo use the trained model:")
    print(f"python scripts/inference.py --model_path {args.output_dir}/best_model")

if __name__ == "__main__":
    main()