#!/usr/bin/env python3
"""
GPT-OSS-20B Training Script
Following official NeMo documentation:
https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/gpt_oss.html

GPT-OSS-20B has 32 experts with topk=4 by default (configurable 1-32)
"""

from nemo.collections import llm
import nemo_run as run

if __name__ == "__main__":
    # Create model configuration
    # GPT-OSS-20B: 32 experts, topk=4 (default)
    config = llm.GPTOSSConfig20B()

    # Optional: Adjust MoE routing (1-32 experts)
    # config.moe_router_topk = 2  # Use 2 experts (faster, lower quality)
    # config.moe_router_topk = 4  # Default: 4 experts (balanced)
    # config.moe_router_topk = 8  # Use 8 experts (slower, higher quality)

    print(f"Model: GPT-OSS-20B")
    print(f"Total experts: {config.num_moe_experts}")
    print(f"Active experts per token: {config.moe_router_topk}")

    # Import from OpenAI format
    llm.import_ckpt(
        model=llm.GPTOSSModel(config),
        source='openai:///workspace/checkpoints/gpt-oss-20b'
    )

    # Configure fine-tuning recipe
    recipe = llm.gpt_oss_20b.finetune_recipe(
        name="gpt_oss_20b_finetuning",
        dir="/workspace/checkpoints",
        num_nodes=1,
        num_gpus_per_node=1,  # RTX 3090 (24GB)
        peft_scheme='lora'  # LoRA for memory efficiency
    )

    # Run training
    run.run(recipe, executor=run.LocalExecutor())