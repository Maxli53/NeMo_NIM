#!/usr/bin/env python3
"""
Simple chat interface for testing the fine-tuned GPT-OSS-20B model
Uses the official Unsloth chat template approach
https://docs.unsloth.ai/basics/chat-templates
"""

import os
import sys
import torch
from unsloth import FastLanguageModel
import time

def load_model(model_path="models/latest", gpu_id=1):
    """
    Load the fine-tuned model with LoRA adapters
    https://github.com/unslothai/unsloth
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print("Loading GPT-OSS-20B model...")
    # Always load base model first (4-bit quantized)
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

    # Enable fast inference mode
    FastLanguageModel.for_inference(model)
    print("Model ready!\n")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, reasoning="medium", max_tokens=150):
    """
    Generate a response using the official tokenizer.apply_chat_template
    This ensures consistency with training format
    """

    # Create messages in the standard format
    # The system message can be customized (model_identity parameter in template)
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]

    # Apply the official chat template
    # This uses the chat_template.jinja saved with the model
    # Parameters match what the template expects (see template header)
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,              # Return string, not tokens
        add_generation_prompt=True,  # Add <|start|>assistant for generation
        reasoning_effort=reasoning,  # GPT-OSS specific: low/medium/high
    )

    # Tokenize for model input
    inputs = tokenizer([formatted], return_tensors="pt").to("cuda")

    print(f"\nGenerating response (reasoning: {reasoning})...")
    start = time.time()

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,  # Use proper EOS token
        )

    elapsed = time.time() - start

    # Decode the full output
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract only the assistant's response
    # The model generates: <|start|>assistant<|channel|>CHANNEL<|message|>CONTENT<|return|>
    # We want just the CONTENT part
    assistant_response = ""

    if "<|start|>assistant" in full_response:
        # Split at assistant marker
        assistant_part = full_response.split("<|start|>assistant")[-1]

        # Handle different channel types (analysis, commentary, final)
        if "<|channel|>" in assistant_part:
            # Extract content after channel and message markers
            parts = assistant_part.split("<|message|>")
            if len(parts) > 1:
                content = parts[1]
                # Clean up end markers
                for marker in ["<|return|>", "<|end|>", "<|endoftext|>"]:
                    if marker in content:
                        content = content.split(marker)[0]
                assistant_response = content.strip()
        else:
            # Fallback for simpler format
            if "<|message|>" in assistant_part:
                content = assistant_part.split("<|message|>")[1]
                for marker in ["<|return|>", "<|end|>", "<|endoftext|>"]:
                    if marker in content:
                        content = content.split(marker)[0]
                assistant_response = content.strip()

    # If we still don't have a response, try simpler extraction
    if not assistant_response and len(outputs[0]) > len(inputs.input_ids[0]):
        # Just decode the new tokens
        new_tokens = outputs[0][len(inputs.input_ids[0]):]
        assistant_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    tokens = len(outputs[0]) - len(inputs.input_ids[0])
    tps = tokens/elapsed if elapsed > 0 else 0

    return assistant_response, tokens, elapsed, tps

def main():
    """Main entry point for the chat interface"""

    # Parse command line arguments
    if len(sys.argv) > 1:
        # Join all arguments as the prompt
        prompt = " ".join(sys.argv[1:])

        # Check for reasoning level flag
        reasoning = "medium"  # Default
        if "--low" in sys.argv:
            reasoning = "low"
            prompt = prompt.replace("--low", "").strip()
        elif "--high" in sys.argv:
            reasoning = "high"
            prompt = prompt.replace("--high", "").strip()

        # Load model - use latest trained model by default
        model, tokenizer = load_model("models/latest")

        # Generate response
        response, tokens, time_taken, tps = generate_response(
            model, tokenizer, prompt, reasoning=reasoning, max_tokens=150
        )

        # Display results
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        print(f"\nResponse: {response}")
        print(f"\n[Generated {tokens} tokens in {time_taken:.2f}s = {tps:.1f} tokens/sec]")

    else:
        # Show usage instructions
        print("Usage: python scripts/chat.py 'Your question here' [--low|--high]")
        print("\nExample prompts to try:")
        print("  python scripts/chat.py 'What is machine learning?'")
        print("  python scripts/chat.py 'Explain quantum computing simply' --low")
        print("  python scripts/chat.py 'Write a Python function for factorial' --high")
        print("  python scripts/chat.py 'What makes a good programmer?'")
        print("\nReasoning levels:")
        print("  --low    : Quick, direct answers")
        print("  --high   : Detailed reasoning with analysis")
        print("  (default): Balanced responses (medium)")
        print("\nNote: This uses the official GPT-OSS-20B chat template")
        print("See: https://github.com/unslothai/unsloth")

if __name__ == "__main__":
    main()