#!/usr/bin/env python3
"""
Export Unsloth fine-tuned model to llama.cpp GGUF format
Based on official documentation: https://docs.unsloth.ai/
"""

import os
import argparse
import subprocess
from unsloth import FastLanguageModel
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Export Unsloth model for llama.cpp")

    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to fine-tuned model")
    parser.add_argument("--output_dir", type=str, default="./exports",
                       help="Output directory for exports")
    parser.add_argument("--merge_method", type=str, default="16bit",
                       choices=["16bit", "4bit"],
                       help="Merge method for the model")
    parser.add_argument("--quantization", type=str, default="Q8_0",
                       choices=["Q2_K", "Q3_K_S", "Q3_K_M", "Q4_0", "Q4_1",
                               "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M",
                               "Q6_K", "Q8_0", "F16"],
                       help="GGUF quantization level")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push merged model to HuggingFace Hub")
    parser.add_argument("--hub_repo", type=str,
                       help="HuggingFace Hub repository name")
    parser.add_argument("--hf_token", type=str,
                       help="HuggingFace token for pushing to hub")
    parser.add_argument("--build_llama_cpp", action="store_true",
                       help="Build llama.cpp if not available")

    return parser.parse_args()

def load_model(args):
    """Load the fine-tuned model"""
    print(f"Loading model from: {args.model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=16384,
        dtype=None,
        load_in_4bit=True if args.merge_method == "4bit" else False,
    )

    print("Model loaded successfully!")
    return model, tokenizer

def merge_and_save(model, tokenizer, args):
    """Merge LoRA adapters and save model"""

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.push_to_hub and args.hub_repo:
        # Push directly to HuggingFace Hub
        print(f"\nPushing merged model to HuggingFace Hub: {args.hub_repo}")

        model.push_to_hub_merged(
            args.hub_repo,
            tokenizer=tokenizer,
            token=args.hf_token,
            save_method=f"merged_{args.merge_method}"
        )

        print(f"Model pushed to: https://huggingface.co/{args.hub_repo}")

    else:
        # Save locally
        merge_dir = os.path.join(args.output_dir, f"merged_{args.merge_method}")

        print(f"\nMerging and saving model to: {merge_dir}")

        # Save merged model (official method from docs)
        model.save_pretrained_merged(
            merge_dir,
            tokenizer,
            save_method=f"merged_{args.merge_method}"
        )

        print(f"Merged model saved to: {merge_dir}")
        return merge_dir

def build_llama_cpp(args):
    """Build llama.cpp with CUDA support"""

    llama_cpp_path = Path("./llama.cpp")

    if llama_cpp_path.exists():
        print("llama.cpp already exists")
        return str(llama_cpp_path)

    print("\nBuilding llama.cpp...")

    commands = [
        "apt-get update",
        "apt-get install -y pciutils build-essential cmake curl libcurl4-openssl-dev",
        "git clone https://github.com/ggml-org/llama.cpp",
        "cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON",
        "cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split llama-quantize",
        "cp llama.cpp/build/bin/llama-* llama.cpp/"
    ]

    for cmd in commands:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            if "apt-get" not in cmd:  # Don't fail on apt-get errors (might need sudo)
                raise RuntimeError(f"Failed to execute: {cmd}")

    print("llama.cpp built successfully!")
    return str(llama_cpp_path)

def convert_to_gguf(merge_dir, args):
    """Convert merged model to GGUF format"""

    print("\nConverting to GGUF format...")

    # Check if llama.cpp exists
    llama_cpp_path = Path("./llama.cpp")

    if not llama_cpp_path.exists():
        if args.build_llama_cpp:
            llama_cpp_path = build_llama_cpp(args)
        else:
            print("llama.cpp not found. Use --build_llama_cpp to build it automatically")
            return

    # Paths
    gguf_output = os.path.join(args.output_dir, "model.gguf")
    quantized_output = os.path.join(args.output_dir, f"model-{args.quantization}.gguf")

    # Convert to GGUF
    convert_cmd = f"python3 {llama_cpp_path}/convert_hf_to_gguf.py {merge_dir} --outfile {gguf_output}"
    print(f"Running: {convert_cmd}")

    result = subprocess.run(convert_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Conversion error: {result.stderr}")
        print("\nTrying alternative conversion method...")

        # Try using Unsloth's built-in GGUF export
        model, tokenizer = load_model(args)
        model.save_pretrained_gguf(
            args.output_dir,
            tokenizer,
            quantization_method=args.quantization.lower().replace("_", "-")
        )
        print(f"GGUF model saved using Unsloth method to: {args.output_dir}")
        return

    print(f"GGUF model created: {gguf_output}")

    # Quantize if not F16
    if args.quantization != "F16":
        quantize_cmd = f"{llama_cpp_path}/llama-quantize {gguf_output} {quantized_output} {args.quantization}"
        print(f"Running: {quantize_cmd}")

        result = subprocess.run(quantize_cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Quantized model created: {quantized_output}")

            # Print size comparison
            original_size = os.path.getsize(gguf_output) / (1024**3)
            quantized_size = os.path.getsize(quantized_output) / (1024**3)
            print(f"\nSize comparison:")
            print(f"  Original: {original_size:.2f} GB")
            print(f"  Quantized: {quantized_size:.2f} GB")
            print(f"  Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
        else:
            print(f"Quantization error: {result.stderr}")

def print_usage_instructions(args):
    """Print instructions for using the exported model"""

    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)

    print("\nTo use with llama.cpp:")

    model_file = f"model-{args.quantization}.gguf" if args.quantization != "F16" else "model.gguf"
    model_path = os.path.join(args.output_dir, model_file)

    print(f"""
./llama.cpp/llama-cli --model {model_path} \\
    --jinja -ngl 99 --threads -1 --ctx-size 16384 \\
    --temp 1.0 --top-p 1.0 --top-k 0 \\
    --conversation
    """)

    print("\nFor single prompt inference:")
    print(f"""
./llama.cpp/llama-cli --model {model_path} \\
    --jinja -ngl 99 --threads -1 --ctx-size 16384 \\
    --temp 1.0 --top-p 1.0 --top-k 0 \\
    -p "Your prompt here"
    """)

    print("\nRecommended quantization levels:")
    print("  Q4_K_M: Best quality/size ratio (~12GB)")
    print("  Q8_0: Near lossless (~13GB)")
    print("  F16: Full quality (~14GB)")

def main():
    args = parse_args()

    print("=" * 60)
    print("Unsloth to llama.cpp Export Tool")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model(args)

    # Merge and save
    merge_dir = merge_and_save(model, tokenizer, args)

    # Convert to GGUF if saved locally
    if merge_dir and not args.push_to_hub:
        convert_to_gguf(merge_dir, args)
        print_usage_instructions(args)

    print("\nExport process complete!")

if __name__ == "__main__":
    main()