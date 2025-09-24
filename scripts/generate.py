#!/usr/bin/env python3
"""
Generate text using GPT-OSS-20B NeMo model
Based on official NeMo inference documentation
"""

import argparse
import torch
from pathlib import Path
from nemo.collections.nlp.models.language_modeling import MegatronGPTModel
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam, SamplingParam
)


class GPTOSSGenerator:
    """Generator class for GPT-OSS models."""

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the generator.

        Args:
            model_path: Path to the .nemo model file
            device: Device to run on (cuda/cpu)
        """
        print(f"Loading model from {model_path}...")
        self.model = MegatronGPTModel.restore_from(model_path)
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device
        print("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        min_length: int = 1,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
    ):
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            min_length: Minimum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty factor
            do_sample: Whether to use sampling or greedy decoding

        Returns:
            Generated text string
        """
        # Configure length parameters
        length_params = LengthParam(
            max_length=max_length,
            min_length=min_length,
        )

        # Configure sampling parameters
        sampling_params = SamplingParam(
            use_greedy=not do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            add_BOS=False,
            all_probs=False,
            compute_logprob=False,
        )

        # Generate text
        with torch.no_grad():
            output = self.model.generate(
                inputs=[prompt],
                length_params=length_params,
                sampling_params=sampling_params,
            )

        return output[0]

    def batch_generate(self, prompts: list, **kwargs):
        """Generate text for multiple prompts."""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description="Generate text with GPT-OSS")

    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the .nemo model file"
    )

    # Generation parameters
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="File containing prompts (one per line)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="Minimum generation length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 = deterministic)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty factor"
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )

    # Output configuration
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="File to save generated text"
    )

    args = parser.parse_args()

    # Validate input
    if args.prompt is None and args.prompt_file is None:
        parser.error("Either --prompt or --prompt-file must be specified")

    # Initialize generator
    generator = GPTOSSGenerator(args.model_path)

    # Get prompts
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [args.prompt]

    # Generate text
    print("\nGenerating text...")
    print("=" * 50)

    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        print("-" * 30)

        generated = generator.generate(
            prompt=prompt,
            max_length=args.max_length,
            min_length=args.min_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.greedy,
        )

        print(f"Generated: {generated}")
        results.append({"prompt": prompt, "generated": generated})

    # Save output if requested
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for result in results:
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Generated: {result['generated']}\n")
                f.write("-" * 50 + "\n")

        print(f"\nOutput saved to {output_path}")


if __name__ == "__main__":
    main()