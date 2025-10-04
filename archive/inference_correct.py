#!/usr/bin/env python3
"""
Correct Unsloth inference approach for fine-tuned GPT-OSS-20B
Following the exact pattern from training script that WORKS
"""

import os
import torch
from unsloth import FastLanguageModel

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

print("Loading base model and applying LoRA adapter...")

# Step 1: Load the base model (same as training)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",  # Base model
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

# Step 2: Load the LoRA adapter
# The adapter was saved to final_model/
from peft import PeftModel
print("Loading LoRA adapter from final_model/...")
model = PeftModel.from_pretrained(model, "final_model/")

# Step 3: Enable fast inference (same as in training script)
FastLanguageModel.for_inference(model)
print("Model ready for inference!")

# Step 4: Test inference with proper chat template
def generate_response(prompt, reasoning_effort="medium", max_new_tokens=100):
    # Use the exact chat template format from training
    formatted_prompt = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Reasoning: {reasoning_effort}<|end|>
<|start|>user<|message|>{prompt}<|end|>
<|start|>assistant<|channel|>"""

    inputs = tokenizer(
        [formatted_prompt],
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to("cuda")

    print(f"\nPrompt: {prompt}")
    print(f"Reasoning effort: {reasoning_effort}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract just the assistant's response
    if "<|start|>assistant<|channel|>" in response:
        response = response.split("<|start|>assistant<|channel|>")[-1]
        # Remove any end tokens
        if "<|end|>" in response:
            response = response.split("<|end|>")[0]

    return response.strip()

# Test with various prompts
print("\n" + "="*60)
print("Testing inference with fine-tuned model")
print("="*60)

test_prompts = [
    "What is 2+2?",
    "Explain machine learning in one sentence.",
    "What is the capital of France?",
    "Write a haiku about coding.",
]

for prompt in test_prompts:
    response = generate_response(prompt, reasoning_effort="low", max_new_tokens=50)
    print(f"Response: {response}\n")
    print("-"*40)

# Test different reasoning efforts
print("\nTesting different reasoning efforts with same prompt:")
test_prompt = "Explain what makes a good programmer."

for effort in ["low", "medium", "high"]:
    response = generate_response(test_prompt, reasoning_effort=effort, max_new_tokens=100)
    print(f"\nReasoning {effort}: {response}")
    print("-"*40)

print("\nInference complete!")