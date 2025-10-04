#!/usr/bin/env python3
"""
Debug script to investigate why GPT-OSS-20B final channel cuts off
Tests various hypotheses about token generation and channel handling
"""

import os
import sys
import torch
from unsloth import FastLanguageModel
from peft import PeftModel
import json

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load_model(use_finetuned=False):
    """Load model with optional fine-tuned adapter"""
    print(f"\n{'='*60}")
    print(f"Loading {'fine-tuned' if use_finetuned else 'base'} model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    if use_finetuned and os.path.exists("models/latest"):
        print("Applying LoRA adapter...")
        model = PeftModel.from_pretrained(model, "models/latest")

    FastLanguageModel.for_inference(model)
    return model, tokenizer

def analyze_tokens(tokenizer):
    """Analyze special tokens and their IDs"""
    print("\n" + "="*60)
    print("SPECIAL TOKENS ANALYSIS:")
    print("="*60)

    special_tokens = {
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
        "bos_token": tokenizer.bos_token,
        "unk_token": tokenizer.unk_token,
    }

    for name, token in special_tokens.items():
        if token:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"{name:15} = '{token}' (ID: {token_id})")

    # Check channel markers
    channel_markers = [
        "<|channel|>", "<|message|>", "<|return|>",
        "analysis<|message|>", "commentary<|message|>", "final<|message|>",
        "<|start|>", "<|end|>", "<|endoftext|>"
    ]

    print("\nCHANNEL MARKERS:")
    for marker in channel_markers:
        try:
            tokens = tokenizer.encode(marker, add_special_tokens=False)
            print(f"'{marker}' -> IDs: {tokens}")
        except:
            print(f"'{marker}' -> Cannot encode")

    return tokenizer.eos_token_id

def test_generation_configs(model, tokenizer, prompt="What date were you trained on?"):
    """Test different generation configurations"""
    print("\n" + "="*60)
    print("TESTING GENERATION CONFIGURATIONS")
    print("="*60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    # Test different reasoning efforts
    for reasoning in ["low", "medium", "high"]:
        print(f"\n--- Reasoning: {reasoning} ---")

        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=reasoning,
            )
        except:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = tokenizer([formatted], return_tensors="pt").to("cuda")

        # Test 1: No EOS token (let it generate freely)
        print("\nTest 1: No EOS token specified")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                eos_token_id=None,  # Disable EOS
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=False)
        print(f"Generated (first 500 chars): {generated[:500]}")

        # Count channel occurrences
        if "analysis<|message|>" in generated:
            print("✓ Has analysis channel")
        if "commentary<|message|>" in generated:
            print("✓ Has commentary channel")
        if "final<|message|>" in generated:
            print("✓ Has final channel")
            # Extract final channel content
            final_parts = generated.split("final<|message|>")
            if len(final_parts) > 1:
                final_content = final_parts[-1].split("<|")[0] if "<|" in final_parts[-1] else final_parts[-1]
                print(f"Final channel content: '{final_content[:100]}'")

        # Test 2: With custom stopping criteria
        print("\nTest 2: Custom EOS token list")
        eos_tokens = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end|>") if "<|end|>" in tokenizer.get_vocab() else tokenizer.eos_token_id]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                eos_token_id=eos_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=False)
        print(f"Generated (first 500 chars): {generated[:500]}")

def decode_token_by_token(model, tokenizer, prompt="What is 2+2?"):
    """Generate and decode token by token to see exactly what happens"""
    print("\n" + "="*60)
    print("TOKEN-BY-TOKEN GENERATION")
    print("="*60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="medium",
    )

    inputs = tokenizer([formatted], return_tensors="pt").to("cuda")

    print(f"\nPrompt: {prompt}")
    print("Generating tokens one by one...")
    print("-" * 40)

    # Generate up to 100 tokens
    generated_ids = inputs.input_ids[0].tolist()

    with torch.no_grad():
        for i in range(100):
            # Get next token
            current_input = torch.tensor([generated_ids]).to("cuda")
            outputs = model(current_input)
            logits = outputs.logits[0, -1, :]

            # Sample next token
            probs = torch.softmax(logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Decode and print
            token_str = tokenizer.decode([next_token], skip_special_tokens=False)
            print(f"Token {i:3d}: ID={next_token:6d} -> '{token_str}'", end="")

            # Check if we hit final channel
            generated_ids.append(next_token)
            full_text = tokenizer.decode(generated_ids[len(inputs.input_ids[0]):], skip_special_tokens=False)

            if "final<|message|>" in full_text and "final<|message|>" in full_text[-20:]:
                print("\n*** ENTERED FINAL CHANNEL ***")

            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                print(f"\n*** HIT EOS TOKEN (ID: {tokenizer.eos_token_id}) ***")
                break
            elif next_token == 200002:  # The <|return|> token
                print(f"\n*** HIT <|return|> TOKEN (ID: 200002) ***")
                break
            else:
                print()

            # Stop if we see certain markers
            if any(marker in token_str for marker in ["<|return|>", "<|end|>", "<|endoftext|>"]):
                print(f"*** STOPPING: Found end marker in token ***")
                break

    print("-" * 40)
    print("\nFull generated text:")
    full_generated = tokenizer.decode(generated_ids[len(inputs.input_ids[0]):], skip_special_tokens=False)
    print(full_generated)

def test_without_template(model, tokenizer):
    """Test raw generation without chat template"""
    print("\n" + "="*60)
    print("RAW GENERATION TEST (No Template)")
    print("="*60)

    # Simple prompt without any template
    prompt = "The capital of France is"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    print(f"Prompt: '{prompt}'")
    print("Generating without chat template...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            eos_token_id=None,  # No EOS
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=False)
    print(f"Generated: {generated}")

def main():
    """Run all debug tests"""
    print("GPT-OSS-20B Channel Debug Script")
    print("=" * 60)

    # Test base model
    print("\n\n### TESTING BASE MODEL ###")
    model, tokenizer = load_model(use_finetuned=False)

    # Analyze tokens
    eos_token_id = analyze_tokens(tokenizer)

    # Test generation configs
    test_generation_configs(model, tokenizer)

    # Token by token analysis
    decode_token_by_token(model, tokenizer)

    # Raw generation test
    test_without_template(model, tokenizer)

    # Test fine-tuned model
    if os.path.exists("models/latest"):
        print("\n\n### TESTING FINE-TUNED MODEL ###")
        model, tokenizer = load_model(use_finetuned=True)
        decode_token_by_token(model, tokenizer, "What date were you trained on?")

    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)
    print("\nKey findings will help identify why final channel cuts off.")
    print("Look for:")
    print("1. Which token ID appears right after 'I' in final channel")
    print("2. Whether it's token 200002 (<|return|>)")
    print("3. If disabling EOS allows full final channel generation")
    print("4. If the issue is template-specific or general")

if __name__ == "__main__":
    main()