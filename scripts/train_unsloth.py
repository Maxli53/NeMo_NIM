#!/usr/bin/env python3
"""
100% Unsloth-Compliant GPT-OSS-20B Training Script
Following official Unsloth documentation and notebooks exactly
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only
import torch

# ============================================================
# STEP 1: Load Pre-Quantized 4-bit Model (Official Unsloth Way)
# ============================================================
max_seq_length = 1024  # Can use 2048 if you have more VRAM
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
    r=8,  # Official recommendation
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # All 7 layers
    lora_alpha=16,  # 2x rank ratio
    lora_dropout=0,  # 0 for speed
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% less VRAM
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ============================================================
# STEP 3: Load and Prepare Dataset (Simple Unsloth Way)
# ============================================================
dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train[:1000]")
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
        # Core settings
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # Effective batch size = 16

        # Learning settings
        learning_rate=2e-4,
        num_train_epochs=1,
        max_steps=30,  # Quick test

        # Optimizer
        optim="adamw_8bit",
        weight_decay=0.01,

        # Scheduler
        lr_scheduler_type="linear",
        warmup_steps=10,

        # Precision (RTX 3090)
        fp16=False,
        bf16=True,

        # Logging
        logging_steps=1,

        # Saving
        output_dir="outputs",
        save_strategy="steps",
        save_steps=15,

        # Other
        seed=3407,
        max_seq_length=max_seq_length,
        packing=False,
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
print("Starting 100% Unsloth-Compliant Training")
print("=" * 60)
print(f"Model: unsloth/gpt-oss-20b-unsloth-bnb-4bit")
print(f"LoRA: r=8, alpha=16 (2:1 ratio)")
print(f"Dataset: {len(dataset)} samples")
print(f"Batch size: 2 x 8 = 16 effective")
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
model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"Final loss: {trainer_stats.training_loss:.4f}")
print(f"Model saved to: final_model/")

# Show inference example
FastLanguageModel.for_inference(model)
inputs = tokenizer(
    "What is 2+2?",
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=20)
print(f"\nTest inference: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")