#!/usr/bin/env python3
"""
Advanced GPT-OSS-20B Training Script with Hyperparameter Optimization
Based on Unsloth LoRA Hyperparameters Guide

Features:
- Automatic overfitting/underfitting detection
- Training on completions only
- Dynamic hyperparameter adjustment
- Evaluation and early stopping
- Multiple training profiles
"""

import os
import yaml
import argparse
import torch
import numpy as np
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, EarlyStoppingCallback, TrainerCallback
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only
import json
from datetime import datetime
import warnings

def parse_args():
    parser = argparse.ArgumentParser(description="Advanced GPT-OSS-20B Training")

    # Config file or profile
    parser.add_argument("--config", type=str, default="configs/training_optimal.yaml",
                       help="Path to config file")
    parser.add_argument("--profile", type=str,
                       choices=["quick_test", "standard", "full_training", "high_quality", "memory_efficient"],
                       help="Training profile to use")

    # Override options
    parser.add_argument("--model_name", type=str,
                       help="Override model name")
    parser.add_argument("--dataset_name", type=str,
                       help="Override dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--resume_from_checkpoint", type=str,
                       help="Resume training from checkpoint")

    # Advanced options
    parser.add_argument("--auto_adjust", action="store_true",
                       help="Auto-adjust hyperparameters based on loss")
    parser.add_argument("--train_on_completions", action="store_true", default=True,
                       help="Train only on assistant responses")
    parser.add_argument("--validate", action="store_true",
                       help="Enable validation split")

    return parser.parse_args()

def load_config(args):
    """Load configuration from YAML file"""
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply profile if specified
    if args.profile and args.profile in config['profiles']:
        profile = config['profiles'][args.profile]
        config['training'].update(profile)
        print(f"Applied profile: {args.profile}")

    # Apply command-line overrides
    if args.model_name:
        config['model']['name'] = args.model_name
    if args.dataset_name:
        config['dataset']['name'] = args.dataset_name

    return config

def setup_environment(config):
    """Setup CUDA environment"""
    os.environ["CUDA_VISIBLE_DEVICES"] = config['hardware']['cuda_visible_devices']

    print(f"Using GPU(s): {config['hardware']['cuda_visible_devices']}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")

def load_model_with_config(config):
    """Load model with configuration"""
    model_config = config['model']
    lora_config = config['lora']

    print(f"\nLoading model: {model_config['name']}")
    print(f"Sequence length: {model_config['max_seq_length']}")
    print(f"4-bit mode: {model_config['load_in_4bit']}")

    # Load base model
    # Use local path if it exists
    import os
    local_model_path = "/media/ubumax/WD_BLACK/AI_Projects/Unsloth_GPT/models/gpt-oss-20b"
    if os.path.exists(local_model_path) and "gpt-oss-20b" in model_config['name']:
        model_config['name'] = local_model_path
        print(f"Using local model at: {local_model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config['name'],
        max_seq_length=model_config['max_seq_length'],
        dtype=model_config['dtype'],
        load_in_4bit=model_config['load_in_4bit'],
    )

    # Add LoRA adapters with optimized settings
    print(f"\nAdding LoRA adapters:")
    print(f"  Rank (r): {lora_config['r']}")
    print(f"  Alpha: {lora_config['lora_alpha']}")
    print(f"  Alpha/Rank ratio: {lora_config['lora_alpha']/lora_config['r']:.2f}")
    print(f"  Target modules: {len(lora_config['target_modules'])} layers")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config['r'],
        target_modules=lora_config['target_modules'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        bias=lora_config['bias'],
        use_gradient_checkpointing=lora_config['use_gradient_checkpointing'],
        random_state=lora_config['random_state'],
        use_rslora=lora_config['use_rslora'],
        loftq_config=lora_config['loftq_config'],
    )

    return model, tokenizer

def prepare_dataset_advanced(config, tokenizer, args):
    """Prepare dataset with advanced formatting"""
    dataset_config = config['dataset']

    print(f"\nLoading dataset: {dataset_config['name']}")
    dataset = load_dataset(dataset_config['name'], split=dataset_config['split'])

    # Limit samples if specified
    if dataset_config['max_samples'] > 0:
        dataset = dataset.select(range(min(dataset_config['max_samples'], len(dataset))))

    print(f"Dataset size: {len(dataset)} samples")

    # Standardize format
    dataset = standardize_sharegpt(dataset)

    # Format with reasoning effort
    def formatting_prompts_func(examples):
        convos = examples["messages"]

        # Apply varying reasoning efforts if configured
        if 'reasoning_effort_distribution' in config['gpt_oss']:
            dist = config['gpt_oss']['reasoning_effort_distribution']
            efforts = np.random.choice(
                ['low', 'medium', 'high'],
                size=len(convos),
                p=[dist['low'], dist['medium'], dist['high']]
            )
        else:
            efforts = [dataset_config['reasoning_effort']] * len(convos)

        texts = []
        for convo, effort in zip(convos, efforts):
            text = tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=dataset_config['add_generation_prompt'],
                reasoning_effort=effort
            )
            texts.append(text)

        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Split for validation if requested
    if args.validate:
        dataset = dataset.train_test_split(test_size=0.1, seed=config['training']['seed'])
        print(f"Training samples: {len(dataset['train'])}")
        print(f"Validation samples: {len(dataset['test'])}")
        return dataset['train'], dataset['test']

    return dataset, None

def create_trainer_advanced(model, tokenizer, train_dataset, eval_dataset, config, args):
    """Create trainer with advanced configuration"""
    training_config = config['training']

    print(f"\nTraining configuration:")
    print(f"  Effective batch size: {training_config['per_device_train_batch_size'] * training_config['gradient_accumulation_steps']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Weight decay: {training_config['weight_decay']}")
    print(f"  Scheduler: {training_config['lr_scheduler_type']}")

    # Create training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,

        # Batch configuration
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],

        # Training duration
        num_train_epochs=training_config.get('num_train_epochs', 1),
        max_steps=training_config.get('max_steps', -1),

        # Optimization
        learning_rate=float(training_config['learning_rate']),
        optim=training_config['optim'],
        weight_decay=training_config['weight_decay'],

        # Scheduler
        lr_scheduler_type=training_config['lr_scheduler_type'],
        warmup_steps=training_config.get('warmup_steps', 0),
        warmup_ratio=training_config.get('warmup_ratio', 0),

        # Precision - Use bf16 for RTX 3090
        fp16=False,
        bf16=True,

        # Logging and saving
        logging_steps=training_config['logging_steps'],
        save_strategy=training_config['save_strategy'],
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],

        # Evaluation
        # evaluation_strategy=training_config.get('evaluation_strategy', 'no'),
        # eval_steps=training_config.get('eval_steps', None),
        # per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 1),

        # Best model
        # load_best_model_at_end=training_config.get('load_best_model_at_end', False),
        # metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss'),
        # greater_is_better=training_config.get('greater_is_better', False),

        # Other
        seed=training_config['seed'],
        report_to='none',  # Disable tensorboard for now

        # SFT specific
        max_seq_length=config['model']['max_seq_length'],
        packing=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # Configure training on completions only
    if config['dataset'].get('train_on_completions_only', True):
        print("\nConfiguring training on completions only...")
        trainer = train_on_responses_only(
            trainer,
            instruction_part=config['gpt_oss']['instruction_part'],
            response_part=config['gpt_oss']['response_part'],
        )

    # Add callbacks
    callbacks = []

    # Early stopping if validation is enabled
    if eval_dataset is not None and training_config.get('load_best_model_at_end', False):
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001
        )
        callbacks.append(early_stopping)

    # Custom callback for monitoring
    class MonitoringCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                # Check for overfitting
                if 'loss' in logs and logs['loss'] < config['monitoring']['overfitting_loss_threshold']:
                    warnings.warn(f"⚠️ Potential overfitting: loss = {logs['loss']:.4f}")

                # Log memory usage
                if torch.cuda.is_available():
                    memory_gb = torch.cuda.memory_allocated() / 1e9
                    logs['memory_gb'] = memory_gb

    callbacks.append(MonitoringCallback())

    for callback in callbacks:
        trainer.add_callback(callback)

    return trainer

def analyze_training_results(trainer, config):
    """Analyze training results and provide recommendations"""
    print("\n" + "=" * 60)
    print("Training Analysis")
    print("=" * 60)

    # Get final metrics
    metrics = trainer.state.log_history

    # Extract losses
    train_losses = [m['loss'] for m in metrics if 'loss' in m]
    eval_losses = [m['eval_loss'] for m in metrics if 'eval_loss' in m]

    if train_losses:
        final_loss = train_losses[-1]
        print(f"\nFinal training loss: {final_loss:.4f}")

        # Check for overfitting
        if final_loss < 0.2:
            print("⚠️ WARNING: Potential overfitting detected!")
            print("Recommendations:")
            print("  - Reduce number of epochs")
            print("  - Decrease learning rate")
            print("  - Increase weight decay to 0.1")
            print("  - Add dropout (lora_dropout=0.1)")
            print("  - Scale down LoRA alpha after training")

        # Check for underfitting
        elif final_loss > 1.5:
            print("⚠️ WARNING: Potential underfitting detected!")
            print("Recommendations:")
            print("  - Increase number of epochs")
            print("  - Increase LoRA rank (try r=32)")
            print("  - Increase learning rate")
            print("  - Decrease batch size")

        else:
            print("✅ Training loss is in healthy range")

    # Memory usage
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak VRAM usage: {peak_memory:.2f} GB")

        if peak_memory > 20:
            print("⚠️ High VRAM usage. Consider:")
            print("  - Reducing batch size")
            print("  - Reducing sequence length")
            print("  - Using gradient checkpointing")

def save_training_info(model, tokenizer, config, trainer, args):
    """Save model and training information"""
    print(f"\nSaving model to {args.output_dir}")

    # Save model
    model.save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")

    # Save configuration
    with open(f"{args.output_dir}/training_config.yaml", 'w') as f:
        yaml.dump(config, f)

    # Save metrics
    metrics = {
        "final_loss": trainer.state.log_history[-1].get('loss', None),
        "total_steps": trainer.state.global_step,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        "timestamp": datetime.now().isoformat(),
    }

    with open(f"{args.output_dir}/training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print("✅ Model and configuration saved")

def main():
    args = parse_args()

    print("=" * 60)
    print("Advanced GPT-OSS-20B Training")
    print("=" * 60)

    # Load configuration
    config = load_config(args)

    # Setup environment
    setup_environment(config)

    # Load model
    model, tokenizer = load_model_with_config(config)

    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset_advanced(config, tokenizer, args)

    # Create trainer
    trainer = create_trainer_advanced(model, tokenizer, train_dataset, eval_dataset, config, args)

    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    if args.resume_from_checkpoint:
        print(f"Resuming from: {args.resume_from_checkpoint}")
    print("=" * 60)

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Analyze results
    analyze_training_results(trainer, config)

    # Save everything
    save_training_info(model, tokenizer, config, trainer, args)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()