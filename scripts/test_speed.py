#!/usr/bin/env python3
"""
Test inference speed without model loading overhead
"""
import os
import time
import torch
from unsloth import FastLanguageModel

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

print("Loading model once...")
start = time.time()

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    'unsloth/gpt-oss-20b-unsloth-bnb-4bit',
    max_seq_length=2048,
    load_in_4bit=True,
)

# Load adapter
from peft import PeftModel
model = PeftModel.from_pretrained(model, 'models/latest')
FastLanguageModel.for_inference(model)

print(f"Model loaded in {time.time() - start:.1f} seconds\n")

# Test different prompts
test_cases = [
    ("What is 2+2?", "low", 30),
    ("Explain gravity", "medium", 100),
    ("Write a haiku", "low", 50),
]

for prompt, reasoning, max_tokens in test_cases:
    print(f"\nPrompt: {prompt}")
    print(f"Reasoning: {reasoning}, Max tokens: {max_tokens}")

    # Format with template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=reasoning,
    )

    inputs = tokenizer([formatted], return_tensors='pt').to('cuda')

    # Warm up
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    # Actual timing
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - start

    tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
    speed = tokens_generated / elapsed

    print(f"  Generated {tokens_generated} tokens in {elapsed:.2f}s = {speed:.1f} tokens/sec")

print("\n" + "="*60)
print("Summary:")
print("- Unsloth optimizations: Active")
print("- Flash Attention 2: Not compatible with PyTorch 2.8 + CUDA 12.8")
print("- Xformers: Available as fallback")
print("- Expected speed for 20B model on RTX 3090: 15-20 tokens/sec")