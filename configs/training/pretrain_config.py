"""
GPT-OSS-20B Pretraining Configuration
Using NeMo-Run for experiment management
Based on: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html
"""

import nemo_run as run
from pathlib import Path

# Base paths
WORKSPACE_DIR = Path("/workspace")
CHECKPOINT_DIR = WORKSPACE_DIR / "checkpoints"
DATA_DIR = WORKSPACE_DIR / "data"
OUTPUT_DIR = WORKSPACE_DIR / "outputs"

# Training configuration
training_config = run.Config(
    name="gpt_oss_20b_pretraining",
    time_limit="24:00:00",
    dependency="singleton",

    # Trainer configuration
    trainer=run.Config(
        num_nodes=1,
        devices=1,  # Single GPU for RTX 3090
        accelerator="gpu",
        precision="bf16-mixed",
        max_epochs=-1,
        max_steps=100000,
        max_time="23:50:00",

        # Validation
        val_check_interval=1000,
        limit_val_batches=50,

        # Checkpointing
        enable_checkpointing=True,
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,

        # Logging
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
    ),

    # Model configuration
    model=run.Config(
        # Batch sizes
        micro_batch_size=1,
        global_batch_size=8,

        # Sequence length
        seq_length=2048,  # Reduced for memory constraints

        # Optimizer
        optim=run.Config(
            name="distributed_fused_adam",
            lr=1e-4,
            weight_decay=0.01,
            betas=[0.9, 0.95],
            eps=1e-8,

            # Learning rate schedule
            sched=run.Config(
                name="CosineAnnealing",
                warmup_steps=500,
                min_lr=1e-5,
            ),
        ),

        # Data configuration
        data=run.Config(
            data_prefix=str(DATA_DIR / "processed"),
            index_mapping_dir=str(DATA_DIR / "tokenized"),
            splits_string="98,2,0",
            num_workers=4,
            dataloader_type="cyclic",
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
        ),
    ),

    # Experiment manager
    exp_manager=run.Config(
        exp_dir=str(OUTPUT_DIR),
        name="gpt_oss_20b",
        create_tensorboard_logger=True,
        create_wandb_logger=False,
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,

        # Checkpoint configuration
        checkpoint_callback_params=run.Config(
            monitor="val_loss",
            save_top_k=5,
            mode="min",
            save_last=True,
            save_on_train_epoch_end=False,
            filename="gpt_oss_20b-{epoch:02d}-{val_loss:.2f}",
        ),
    ),
)

# LoRA fine-tuning configuration
lora_config = run.Config(
    peft_scheme="lora",

    lora=run.Config(
        target_modules=["attention.qkv", "attention.dense", "mlp.fc1", "mlp.fc2"],
        dim=16,
        alpha=32,
        dropout=0.1,
        dropout_position="pre",
    ),
)