import os
from nemo.collections import llm

os.environ['NEMO_MODELS_CACHE'] = '/workspace/checkpoints'

model_config = llm.GPTOSSConfig20B()
model = llm.GPTOSSModel(model_config)

print("Starting conversion from Hugging Face to NeMo format...")
print(f"Model config: {model_config.num_moe_experts} experts, topk={model_config.moe_router_topk}")
print(f"Output will be saved to: {os.environ['NEMO_MODELS_CACHE']}")

llm.import_ckpt(
    model=model,
    source='hf:///models/gpt-oss-20b'
)

print("Conversion completed successfully!")