#!/usr/bin/env python3
"""
Test with the ACTUAL model we downloaded: unsloth/gpt-oss-20b-unsloth-bnb-4bit
"""

from unsloth import FastLanguageModel

# Use the ACTUAL model name we downloaded!
model_name = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"  # BNB 4-bit version

print(f"Loading {model_name}...")
print("This should use our cached model!")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 1024,
    dtype = None,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)
print("✓ Model loaded!")

# Test
inputs = tokenizer("What is 2+2?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(f"Response: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")