#!/usr/bin/env python3
"""
Train GPT-OSS-20B using NeMo Framework
Using NeMo-Run for experiment management
Based on: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html
"""

import argparse
import nemo_run as run
from nemo.collections import llm
from pathlib import Path
import sys

# Add configs to path
sys.path.append(str(Path(__file__).parent.parent / "configs"))
from training.pretrain_config import training_config


def setup_recipe(args):
    """Setup training recipe based on task type."""

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "pretrain":
        # Pretraining from scratch
        recipe = llm.gpt_oss_20b.pretrain_recipe(
            dir=str(checkpoint_dir),
            name=args.experiment_name,
            num_nodes=args.num_nodes,
            num_gpus_per_node=args.num_gpus,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            constant_steps=args.constant_steps,
            fp8=args.fp8,
            tp_comm_overlap=args.tp_comm_overlap,
        )
    elif args.task == "finetune":
        # Fine-tuning with optional LoRA
        recipe = llm.gpt_oss_20b.finetune_recipe(
            dir=str(checkpoint_dir),
            name=args.experiment_name,
            num_nodes=args.num_nodes,
            num_gpus_per_node=args.num_gpus,
            peft_scheme=args.peft_scheme,
            restore_from_path=args.restore_from,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")

    return recipe


def main():
    parser = argparse.ArgumentParser(description="Train GPT-OSS with NeMo")

    # Task configuration
    parser.add_argument(
        "--task",
        type=str,
        choices=["pretrain", "finetune"],
        default="finetune",
        help="Training task type"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="gpt_oss_20b_experiment",
        help="Experiment name for tracking"
    )

    # Hardware configuration
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Number of nodes for training"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs per node"
    )

    # Training configuration
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Warmup steps for learning rate"
    )
    parser.add_argument(
        "--constant-steps",
        type=int,
        default=0,
        help="Steps with constant learning rate"
    )

    # Model configuration
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/workspace/checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--restore-from",
        type=str,
        default=None,
        help="Path to checkpoint to restore from (for fine-tuning)"
    )
    parser.add_argument(
        "--peft-scheme",
        type=str,
        choices=["none", "lora"],
        default="lora",
        help="PEFT method to use (for fine-tuning)"
    )

    # Optimization flags
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Enable FP8 training"
    )
    parser.add_argument(
        "--tp-comm-overlap",
        action="store_true",
        help="Enable tensor parallel communication overlap"
    )

    # Execution configuration
    parser.add_argument(
        "--executor",
        type=str,
        choices=["local", "slurm"],
        default="local",
        help="Execution backend"
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Run directly in the same process (for debugging)"
    )

    args = parser.parse_args()

    # Setup recipe
    recipe = setup_recipe(args)

    # Choose executor
    if args.executor == "local":
        executor = run.LocalExecutor()
    else:
        # Configure SLURM executor if needed
        executor = run.SlurmExecutor(
            nodes=args.num_nodes,
            ntasks_per_node=args.num_gpus,
            time="24:00:00",
            partition="gpu"
        )

    # Run the training
    print(f"Starting {args.task} task: {args.experiment_name}")
    print(f"Using {args.num_nodes} nodes with {args.num_gpus} GPUs each")

    if args.direct:
        # Run directly for debugging
        run.run(recipe, direct=True)
    else:
        # Run with executor
        run.run(recipe, executor=executor)


if __name__ == "__main__":
    main()