#!/usr/bin/env python3
"""
Enhanced Unsloth GPT-OSS-20B Training Script with ALL Optimizations
Targeting 22GB VRAM usage on RTX 3090
Includes: LoftQ, RSLoRA, group_by_length, packing=True
"""

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only
import torch
# Note: Set CUDA_VISIBLE_DEVICES=1 from command line to use GPU 1

# ============================================================
# STEP 1: Load Pre-Quantized 4-bit Model (Official Unsloth Way)
# ============================================================
max_seq_length = 2048  # Increased for 22GB VRAM target
dtype = None  # Auto-detect
load_in_4bit = True  # QLoRA for 14GB VRAM

# Use the ACTUAL pre-quantized model from Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",  # Pre-quantized!
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# ============================================================
# STEP 2: Add LoRA Adapters (Official Settings)
# ============================================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Higher rank for quality (22GB VRAM target)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # All 7 layers
    lora_alpha=32,  # 2x rank ratio (16*2=32)
    lora_dropout=0,  # 0 for speed
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% less VRAM
    random_state=3407,
    use_rslora=True,  # Rank-stabilized for r=16
    loftq_config={  # Better initialization
        "loftq_bits": 4,
        "loftq_iter": 1,
    },
)

# ============================================================
# STEP 3: Load and Prepare Dataset (Simple Unsloth Way)
# ============================================================
dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train[:10000]")  # More data for 22GB target
dataset = standardize_sharegpt(dataset)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = []
    for convo in convos:
        # Simple chat template application
        text = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
            reasoning_effort="medium",  # GPT-OSS feature
        )
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# ============================================================
# STEP 4: Training Arguments (Official Unsloth Settings)
# ============================================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        # Core settings (Optimized for full VRAM usage)
        per_device_train_batch_size=6,  # Maximized for 22GB VRAM
        gradient_accumulation_steps=3,  # Effective batch = 18

        # Learning settings
        learning_rate=2e-4,
        num_train_epochs=1,
        max_steps=500,  # Full training for quality

        # Optimizer
        optim="adamw_8bit",
        weight_decay=0.01,

        # Scheduler
        lr_scheduler_type="linear",
        warmup_steps=20,  # More warmup for r=16 stability

        # Precision (RTX 3090)
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),

        # Speed optimization
        group_by_length=True,  # 5x speedup

        # Logging
        logging_steps=10,

        # Saving
        output_dir="outputs_enhanced",
        save_strategy="steps",
        save_steps=50,  # Save every 50 steps

        # Other
        seed=3407,
        max_seq_length=max_seq_length,
        packing=False,  # Disabled due to tokenization errors
    ),
)

# ============================================================
# STEP 5: Train on Completions Only (Unsloth Feature)
# ============================================================
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start|>user<|message|>",
    response_part="<|start|>assistant<|channel|>",  # Note: <|channel|> not <|message|>
)

# ============================================================
# STEP 6: Start Training
# ============================================================
print("=" * 60)
print("ENHANCED TRAINING (22GB VRAM TARGET)")
print("=" * 60)
print(f"Model: unsloth/gpt-oss-20b-unsloth-bnb-4bit")
print(f"LoRA: r=16, alpha=32 with RSLoRA")
print(f"Dataset: {len(dataset)} samples")
print(f"Batch: 6 x 3 = 18 effective")
print(f"Max steps: 500")
print(f"Optimizations: packing=False (temp fix), group_by_length=True")
print("=" * 60)

# Show GPU memory before training
if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    gpu_stats = torch.cuda.get_device_properties(current_device)
    start_gpu_memory = round(torch.cuda.memory_allocated(current_device) / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU {current_device}: {gpu_stats.name}")
    print(f"Memory: {start_gpu_memory}/{max_memory} GB")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print()

# Train!
trainer_stats = trainer.train()

# ============================================================
# STEP 7: Save Model (Unsloth Way)
# ============================================================
model.save_pretrained("enhanced_model")
tokenizer.save_pretrained("enhanced_model")

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"Final loss: {trainer_stats.training_loss:.4f}")
print(f"Model saved to: enhanced_model/")

# Show inference example
FastLanguageModel.for_inference(model)
inputs = tokenizer(
    "What is 2+2?",
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=20)
print(f"\nTest inference: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")