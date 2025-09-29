# Unsloth GPT-OSS-20B Complete Roadmap

## Executive Summary

Complete implementation roadmap for GPT-OSS-20B using Unsloth, based on official documentation from docs.unsloth.ai and HuggingFace. Optimized for 2x RTX 3090 (24GB VRAM each) with only 14GB required for QLoRA training.

---

## Model Specifications (Official)

### GPT-OSS-20B Architecture
- **Total Parameters**: 21B (20B marketed)
- **Active Parameters**: 3.6B per token (MoE architecture)
- **Experts**: 32 total, 4 active per token
- **Context Window**: 128K tokens (131,072 maximum)
- **License**: Apache 2.0 (commercial use permitted)

### Memory Requirements
- **QLoRA Training**: 14GB VRAM (fits single RTX 3090)
- **BF16 LoRA Training**: 44GB VRAM (requires both GPUs)
- **Inference (4-bit)**: ~12GB VRAM
- **Inference (16-bit)**: ~14GB VRAM

---

## Phase 1: Environment Setup (Day 1)

### 1.1 System Verification
```bash
# Verify CUDA
nvidia-smi
nvcc --version

# Check Python version (3.10+ required)
python3 --version

# Verify available VRAM
nvidia-smi --query-gpu=memory.total --format=csv
```

### 1.2 Unsloth Installation (Official Command)
```bash
# Create virtual environment
python3 -m venv /media/ubumax/WD_BLACK/AI_Projects/Unsloth_GPT/venv
source /media/ubumax/WD_BLACK/AI_Projects/Unsloth_GPT/venv/bin/activate

# Install Unsloth (official command from docs)
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo

# Additional dependencies
pip install --no-deps trl peft accelerate bitsandbytes
```

### 1.3 Verify Installation
```python
# test_install.py
from unsloth import FastLanguageModel
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
print("Unsloth installation successful!")
```

---

## Phase 2: Model Acquisition (Day 1)

### 2.1 Training Model Options

#### Option A: Pre-quantized 4-bit (Recommended)
```python
model_name = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
# Pre-quantized for efficiency
# Size: ~10GB download
# VRAM: 14GB for QLoRA training
```

#### Option B: Base Model
```python
model_name = "unsloth/gpt-oss-20b"
# Will be quantized during loading
# Size: ~40GB download
# VRAM: Same after quantization
```

### 2.2 GGUF Models for Inference
```bash
# Download GGUF models from HuggingFace
huggingface-cli download unsloth/gpt-oss-20b-GGUF \
    --include "gpt-oss-20b-Q4_K_M.gguf" \
    --local-dir models/gpt-oss-20b/gguf/

# Available quantizations:
# Q4_K_M: 11.9 GB (recommended - best quality/size)
# Q4_K_S: 11.5 GB (smaller, slightly lower quality)
# Q8_0: 13.2 GB (near lossless)
# F16: 13.8 GB (full 16-bit)
```

---

## Phase 3: Dataset Preparation (Day 1-2)

### 3.1 Official Recommendation: 75/25 Split
Per Unsloth documentation:
- **75% Reasoning Examples**: Chain-of-thought, step-by-step
- **25% Direct Answers**: Maintain versatility

### 3.2 Multilingual-Thinking Dataset
```python
from datasets import load_dataset

# Official example dataset
dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")

# Dataset structure
# - messages: conversation format
# - reasoning_effort: low/medium/high
# - language: multi-language support
```

### 3.3 Custom Dataset Format
```python
# Required format for Unsloth
chat_template = {
    "messages": [
        {
            "role": "system",
            "content": "You are ChatGPT, a large language model trained by OpenAI.\nReasoning: medium"
        },
        {
            "role": "user",
            "content": "Your question here"
        },
        {
            "role": "assistant",
            "content": "Model response with reasoning"
        }
    ]
}
```

---

## Phase 4: Training Implementation (Day 2-3)

### 4.1 Complete Training Script
```python
# train.py - Based on official notebook
from unsloth import FastLanguageModel, UnslothTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Load model with official settings
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    max_seq_length=16384,  # Minimum recommended
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # Essential for RTX 3090
)

# 2. Add LoRA adapters (official parameters)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # From official notebook
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

# 3. Load dataset
dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")

# 4. Training arguments (optimized for RTX 3090)
args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=1,  # RTX 3090 constraint
    gradient_accumulation_steps=4,  # Effective batch size = 4
    num_train_epochs=1,
    max_steps=100,  # Start with 100, increase as needed
    learning_rate=2e-4,
    fp16=True,  # Use mixed precision
    optim="adamw_8bit",  # Memory efficient optimizer
    warmup_steps=5,
    logging_steps=1,
    save_strategy="steps",
    save_steps=50,
    seed=42,
)

# 5. Create trainer
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=args,
)

# 6. Train
trainer.train()

# 7. Save model
model.save_pretrained("final_checkpoint")
tokenizer.save_pretrained("final_checkpoint")
```

### 4.2 Memory Optimization Settings
```python
# For 14GB VRAM usage (single RTX 3090)
config = {
    "max_seq_length": 16384,  # Don't exceed this
    "batch_size": 1,  # Keep at 1
    "gradient_accumulation": 4,  # Simulate larger batch
    "load_in_4bit": True,  # Essential
    "use_gradient_checkpointing": "unsloth",  # Save memory
}
```

---

## Phase 5: Export Options (Day 3)

### 5.1 Export to GGUF (Recommended)
```python
# export_gguf.py
from unsloth import FastLanguageModel

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="final_checkpoint",
    max_seq_length=16384,
    dtype=None,
    load_in_4bit=True,
)

# Export to GGUF with Q4_K_M quantization
model.save_pretrained_gguf(
    "models/gpt-oss-20b/gguf",
    tokenizer,
    quantization_method="q4_k_m"  # Best quality/size ratio
)

# Available quantizations:
# q4_k_m: Recommended (11.9 GB)
# q4_k_s: Smaller (11.5 GB)
# q8_0: Near lossless (13.2 GB)
```

### 5.2 Merge and Save (16-bit)
```python
# For full quality preservation
model.save_pretrained_merged(
    "models/gpt-oss-20b/merged",
    tokenizer,
    save_method="merged_16bit"
)

# Or push to HuggingFace
model.push_to_hub_merged(
    "your-username/gpt-oss-20b-finetuned",
    tokenizer=tokenizer,
    token="your_hf_token"
)
```

---

## Phase 6: Inference Setup (Day 3-4)

### 6.1 Python Inference (Official Settings)
```python
# inference.py
from unsloth import FastLanguageModel

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="final_checkpoint",
    max_seq_length=16384,
    dtype=None,
    load_in_4bit=True,
)

# Set to inference mode
FastLanguageModel.for_inference(model)

# Official recommended settings
generation_config = {
    "temperature": 1.0,  # Official recommendation
    "top_p": 1.0,  # Official recommendation
    "top_k": 0,  # Or experiment with 100
    "max_new_tokens": 500,
    "do_sample": True,
}

# Generate with proper chat template
prompt = """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Reasoning: medium<|end|>
<|start|>user<|message|>What is 2+2?<|end|>
<|start|>assistant<|channel|>"""

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, **generation_config)
print(tokenizer.batch_decode(outputs))
```

### 6.2 llama.cpp Setup and Inference
```bash
# Build llama.cpp with CUDA
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# Run inference with official settings
./build/bin/llama-cli \
    --model ../models/gpt-oss-20b/gguf/gpt-oss-20b-Q4_K_M.gguf \
    --ctx-size 16384 \
    --n-gpu-layers 99 \
    --temp 1.0 \
    --top-p 1.0 \
    --top-k 0 \
    --threads $(nproc) \
    --conversation
```

---

## Phase 7: Optimization & Benchmarking (Day 4-5)

### 7.1 Performance Metrics
```python
# benchmark.py
import time
import torch
from unsloth import FastLanguageModel

def benchmark_inference(model, tokenizer, num_runs=10):
    prompt = "Explain quantum computing in simple terms."

    times = []
    tokens_generated = []

    for _ in range(num_runs):
        start = time.time()
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=100)
        end = time.time()

        times.append(end - start)
        tokens_generated.append(len(outputs[0]) - len(inputs.input_ids[0]))

    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    tokens_per_second = avg_tokens / avg_time

    print(f"Average tokens/second: {tokens_per_second:.2f}")
    print(f"Average generation time: {avg_time:.2f}s")
    print(f"Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### 7.2 Expected Performance
| Configuration | VRAM Usage | Tokens/Second | Notes |
|--------------|------------|---------------|-------|
| QLoRA Training | 14GB | N/A | Single RTX 3090 |
| 4-bit Inference | 12GB | 15-25 | Q4_K_M quantization |
| 16-bit Inference | 14GB | 10-15 | Full precision |
| llama.cpp (4-bit) | 12GB | 20-30 | Optimized C++ |

---

## Critical Success Factors

### Must-Have Requirements
- ✅ 14GB VRAM minimum (have 24GB)
- ✅ CUDA 12.0+ (have 12.6)
- ✅ Python 3.10+
- ✅ 75/25 reasoning data split
- ✅ Temperature 1.0 for inference

### Performance Targets
- ✅ Training: <14GB VRAM usage
- ✅ Inference: >15 tokens/second
- ✅ Export: GGUF Q4_K_M successful
- ✅ Quality: Coherent multi-turn conversations

### Risk Mitigation
1. **OOM Errors**: Reduce max_seq_length to 8192
2. **Slow Training**: Verify Flash Attention 2 installed
3. **Poor Quality**: Increase training steps to 500+
4. **Export Issues**: Ensure 50GB+ free disk space

---

## Timeline

### Week 1 Milestones
- Day 1: Environment setup, model download
- Day 2: First training run (100 steps)
- Day 3: GGUF export, llama.cpp setup
- Day 4: Inference optimization
- Day 5: Benchmarking complete

### Week 2 Goals
- Custom dataset integration
- Extended training (1000+ steps)
- API server deployment
- Production optimization

---

## Resources

### Official Documentation
- [Unsloth Docs](https://docs.unsloth.ai/)
- [GPT-OSS Guide](https://docs.unsloth.ai/new/gpt-oss-how-to-run-and-fine-tune)
- [GGUF Models](https://huggingface.co/unsloth/gpt-oss-20b-GGUF)
- [Training Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb)

### Hardware Specifications
- GPU: 2x RTX 3090 (24GB each)
- RAM: 62GB + 100GB swap
- CUDA: 12.6
- Driver: 580.65.06

---

**Document Version**: 2.0
**Created**: 2025-09-29
**Status**: Ready for Implementation
**Based on**: Official Unsloth Documentation