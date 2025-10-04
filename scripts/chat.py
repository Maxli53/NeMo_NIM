#!/usr/bin/env python3
"""
Simple chat interface for testing the fine-tuned GPT-OSS-20B model
Run single prompts to test the model's responses
"""

import os
import sys
import torch
from unsloth import FastLanguageModel
import time

def load_model(model_path="final_model", gpu_id=1):
    """Load the fine-tuned model with correct approach"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print("Loading GPT-OSS-20B model...")
    # Always load base model first
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Apply LoRA adapter if specified
    if not model_path.startswith("unsloth/"):
        from peft import PeftModel
        print(f"Applying LoRA adapter from {model_path}...")
        model = PeftModel.from_pretrained(model, model_path)

    FastLanguageModel.for_inference(model)
    print("Model ready!\n")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, reasoning="low", max_tokens=100):
    """Generate a response with timing"""
    formatted = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Reasoning: {reasoning}<|end|>
<|start|>user<|message|>{prompt}<|end|>
<|start|>assistant<|channel|>"""

    inputs = tokenizer([formatted], return_tensors="pt").to("cuda")

    print(f"\nGenerating response (reasoning: {reasoning})...")
    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    elapsed = time.time() - start

    # Extract response
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "<|start|>assistant<|channel|>" in response:
        response = response.split("<|start|>assistant<|channel|>")[-1]
        if "<|end|>" in response:
            response = response.split("<|end|>")[0]

    tokens = len(outputs[0]) - len(inputs.input_ids[0])

    return response.strip(), tokens, elapsed, tokens/elapsed if elapsed > 0 else 0

def main():
    # Parse simple command line args
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])

        # Load model
        model, tokenizer = load_model()

        # Generate response with LOW reasoning for direct answers
        response, tokens, time_taken, tps = generate_response(
            model, tokenizer, prompt, reasoning="low", max_tokens=150
        )

        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        print(f"\nResponse: {response}")
        print(f"\n[Generated {tokens} tokens in {time_taken:.2f}s = {tps:.1f} tokens/sec]")

    else:
        print("Usage: python scripts/chat.py 'Your question here'")
        print("\nExample prompts to try:")
        print("  python scripts/chat.py 'What is machine learning?'")
        print("  python scripts/chat.py 'Explain quantum computing simply'")
        print("  python scripts/chat.py 'Write a haiku about AI'")
        print("  python scripts/chat.py 'What makes a good programmer?'")

if __name__ == "__main__":
    main()