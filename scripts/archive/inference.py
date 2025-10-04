#!/usr/bin/env python3
"""
GPT-OSS-20B Inference Script using Unsloth
Uses official recommended settings from docs.unsloth.ai

Recommended Settings (from official docs):
- Temperature: 1.0
- Top_P: 1.0
- Top_K: 0 (or experiment with 100)
- Min Context: 16,384 tokens
"""

import os
import argparse
import torch
from unsloth import FastLanguageModel
import time
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with GPT-OSS-20B")

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to fine-tuned model or 'unsloth/gpt-oss-20b-unsloth-bnb-4bit'")
    parser.add_argument("--max_seq_length", type=int, default=16384,
                       help="Maximum sequence length (min 16384 recommended)")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                       help="Load model in 4-bit")

    # Generation arguments (official recommendations)
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature (official: 1.0)")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p sampling (official: 1.0)")
    parser.add_argument("--top_k", type=int, default=0,
                       help="Top-k sampling (official: 0 or 100)")
    parser.add_argument("--max_new_tokens", type=int, default=500,
                       help="Maximum tokens to generate")
    parser.add_argument("--do_sample", action="store_true", default=True,
                       help="Use sampling (recommended)")

    # Input arguments
    parser.add_argument("--prompt", type=str,
                       help="Single prompt for generation")
    parser.add_argument("--prompt_file", type=str,
                       help="File containing prompts (one per line)")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive chat mode")

    # Reasoning effort (GPT-OSS specific)
    parser.add_argument("--reasoning_effort", type=str, default="medium",
                       choices=["low", "medium", "high"],
                       help="Reasoning effort level")

    # Output arguments
    parser.add_argument("--output_file", type=str,
                       help="Save outputs to file")
    parser.add_argument("--stream", action="store_true",
                       help="Stream tokens as they're generated")

    # Other arguments
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU to use (0 or 1)")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmark")

    return parser.parse_args()

def load_model(args):
    """Load model for inference - FIXED to properly load LoRA adapters"""
    print(f"Loading model from: {args.model_path}")
    print(f"Max sequence length: {args.max_seq_length}")

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Check if this is a LoRA adapter path or base model
    if args.model_path.startswith("unsloth/") or args.model_path.startswith("meta-llama/"):
        # This is a base model - load directly
        print("Loading base model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=args.load_in_4bit,
        )
    else:
        # This is likely a LoRA adapter - load base model first
        print("Loading base GPT-OSS-20B model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=args.load_in_4bit,
        )

        # Apply LoRA adapter
        print(f"Applying LoRA adapter from {args.model_path}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model_path)

    # Set to inference mode
    FastLanguageModel.for_inference(model)

    print("Model loaded and ready for inference!")
    return model, tokenizer

def format_prompt(prompt, reasoning_effort="medium"):
    """Format prompt with GPT-OSS chat template"""
    formatted = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Reasoning: {reasoning_effort}<|end|>
<|start|>user<|message|>{prompt}<|end|>
<|start|>assistant<|channel|>"""
    return formatted

def generate_response(model, tokenizer, prompt, args):
    """Generate response for a single prompt"""
    # Format prompt
    formatted_prompt = format_prompt(prompt, args.reasoning_effort)

    # Tokenize
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

    # Track time
    start_time = time.time()

    # Generate with official recommended settings
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generation_time = time.time() - start_time

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract only the assistant's response
    if "<|start|>assistant<|channel|>" in generated_text:
        response = generated_text.split("<|start|>assistant<|channel|>")[-1]
        response = response.split("<|end|>")[0].strip()
    else:
        response = generated_text[len(formatted_prompt):].strip()

    # Calculate tokens/second
    num_tokens = len(outputs[0]) - len(inputs.input_ids[0])
    tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0

    return response, tokens_per_second, generation_time

def interactive_chat(model, tokenizer, args):
    """Interactive chat mode"""
    print("\n" + "=" * 60)
    print("Interactive Chat Mode")
    print("Commands: 'quit' to exit, 'clear' to reset conversation")
    print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Settings: temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    print("=" * 60 + "\n")

    conversation_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                conversation_history = []
                print("Conversation cleared.")
                continue
            elif not user_input:
                continue

            # Generate response
            print("\nAssistant: ", end="", flush=True)
            response, tps, gen_time = generate_response(
                model, tokenizer, user_input, args
            )

            print(response)
            print(f"\n[{tps:.1f} tokens/s, {gen_time:.2f}s]")

            # Add to history
            conversation_history.append({"user": user_input, "assistant": response})

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

def benchmark_performance(model, tokenizer, args):
    """Run performance benchmark"""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    test_prompts = [
        "What is 2+2?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate factorial.",
        "What are the benefits of exercise?",
        "How does photosynthesis work?",
    ]

    total_time = 0
    total_tokens = 0
    results = []

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}/{len(test_prompts)}: {prompt[:50]}...")

        response, tps, gen_time = generate_response(
            model, tokenizer, prompt, args
        )

        total_time += gen_time
        tokens_generated = len(tokenizer.encode(response))
        total_tokens += tokens_generated

        results.append({
            "prompt": prompt,
            "response_length": tokens_generated,
            "time": gen_time,
            "tokens_per_second": tps
        })

        print(f"  Time: {gen_time:.2f}s")
        print(f"  Tokens: {tokens_generated}")
        print(f"  Speed: {tps:.1f} tokens/s")

    # Summary
    print("\n" + "-" * 60)
    print("Benchmark Summary:")
    print(f"Total prompts: {len(test_prompts)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Average speed: {total_tokens/total_time:.1f} tokens/s")

    # Memory usage
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM usage: {peak_memory:.2f} GB")

    return results

def process_file(model, tokenizer, args):
    """Process prompts from file"""
    print(f"\nProcessing prompts from: {args.prompt_file}")

    with open(args.prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]

    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"\nProcessing {i}/{len(prompts)}: {prompt[:50]}...")

        response, tps, gen_time = generate_response(
            model, tokenizer, prompt, args
        )

        results.append({
            "prompt": prompt,
            "response": response,
            "tokens_per_second": tps,
            "generation_time": gen_time
        })

        print(f"Generated {len(tokenizer.encode(response))} tokens in {gen_time:.2f}s")

    # Save results if output file specified
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")

    return results

def main():
    args = parse_args()

    print("=" * 60)
    print("GPT-OSS-20B Inference with Unsloth")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model(args)

    # Print settings
    print(f"\nGeneration settings:")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-P: {args.top_p}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Max tokens: {args.max_new_tokens}")
    print(f"  Reasoning effort: {args.reasoning_effort}")

    # Run appropriate mode
    if args.benchmark:
        benchmark_performance(model, tokenizer, args)
    elif args.interactive:
        interactive_chat(model, tokenizer, args)
    elif args.prompt_file:
        process_file(model, tokenizer, args)
    elif args.prompt:
        print(f"\nPrompt: {args.prompt}\n")
        print("Generating response...\n")

        response, tps, gen_time = generate_response(
            model, tokenizer, args.prompt, args
        )

        print(f"Response:\n{response}\n")
        print(f"\n[Generated in {gen_time:.2f}s at {tps:.1f} tokens/s]")

        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump({
                    "prompt": args.prompt,
                    "response": response,
                    "tokens_per_second": tps,
                    "generation_time": gen_time
                }, f, indent=2)
    else:
        print("\nNo input specified. Use --prompt, --prompt_file, --interactive, or --benchmark")

if __name__ == "__main__":
    main()