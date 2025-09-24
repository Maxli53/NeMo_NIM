#!/usr/bin/env python3
"""
Import OpenAI GPT-OSS-20B Model to NeMo Format
Based on: https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/gpt_oss.html
"""

import argparse
import logging
from pathlib import Path
from nemo.collections import llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def import_gpt_oss_model(source_path: str, output_path: str, model_size: str = "20b"):
    """
    Import GPT-OSS model from OpenAI checkpoint to NeMo format.

    Args:
        source_path: Path to OpenAI checkpoint
        output_path: Path to save the converted NeMo model
        model_size: Model size - "20b" or "120b"
    """
    logger.info(f"Importing GPT-OSS-{model_size.upper()} from {source_path}")

    # Select appropriate config based on model size
    if model_size == "20b":
        model_config = llm.GPTOSSConfig20B()
        model = llm.GPTOSSModel(model_config)
    elif model_size == "120b":
        model_config = llm.GPTOSSConfig120B()
        model = llm.GPTOSSModel(model_config)
    else:
        raise ValueError(f"Unsupported model size: {model_size}")

    # Import from OpenAI format
    logger.info("Starting model import...")
    llm.import_ckpt(
        model=model,
        source=f"openai://{source_path}"
    )

    # Save the converted model
    output_file = Path(output_path) / f"gpt_oss_{model_size}.nemo"
    logger.info(f"Saving converted model to {output_file}")
    model.save_to(str(output_file))

    logger.info("Model import completed successfully!")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Import GPT-OSS model to NeMo format")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to OpenAI GPT-OSS checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/checkpoints/converted",
        help="Output directory for NeMo model"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["20b", "120b"],
        default="20b",
        help="Model size to import"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Import the model
    import_gpt_oss_model(args.source, args.output, args.model_size)


if __name__ == "__main__":
    main()