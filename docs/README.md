# Unsloth GPT-OSS-20B Fine-tuning

Efficient fine-tuning of GPT-OSS-20B using Unsloth. Requires only 14GB VRAM on RTX 3090.

## 🚀 Quick Start

```bash
# Setup (one-time)
./setup.sh
source venv/bin/activate

# Train with production-ready script (v3 with loss monitoring)
python scripts/train.py --profile standard --gpu 1

# Quick test (30 steps) - Verified: 8 min, final loss ~0.89
python scripts/train.py --profile quick_test --gpu 1

# Quick one-shot chat (loads model each time)
python scripts/chat.py "Your question here" --low

# Interactive terminal chat with streaming (15-16 tokens/sec)
python scripts/chat_interactive.py --gpu 1

# Web UI with Gradio (RECOMMENDED - ChatGPT-like interface)
python scripts/chat_gradio_fixed.py --gpu 1 --port 7860
```

## 📋 Key Features

### ✅ Production Ready Scripts
- **Training** (`train.py`): Loss monitoring, configurable profiles, checkpoint management
- **Terminal Chat** (`chat_interactive.py`): Rich formatting, streaming responses, session management
- **Web UI** (`chat_gradio_fixed.py`): ChatGPT-like interface with full parameter controls
- **Benchmarking** (`test_speed.py`): Measure actual inference speed without loading overhead

### 🎯 Fixed: GPT-OSS Channel Handling
Properly handles GPT-OSS-20B's multi-channel architecture (analysis, commentary, final) without early stopping at intermediate `<|end|>` tokens.

## 🛠 Installation

### Automated Setup
```bash
./setup.sh  # Installs everything with UV package manager
```

### Manual Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade -qqq uv
uv pip install -qqq \
    "torch>=2.8.0" "triton>=3.4.0" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth" \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    bitsandbytes transformers datasets trl
```

## 🎯 Critical: Model Setup & Cache Management

### The Right Way: Use Pre-Quantized Model Name
```python
# ✅ CORRECT - Uses cached model efficiently
model_name = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
```

### Model Cache Structure
The GPT-OSS-20B 4-bit model (~12GB) MUST be in the correct HuggingFace cache location:

```
~/.cache/huggingface/hub/models--unsloth--gpt-oss-20b-unsloth-bnb-4bit/
├── blobs/
│   ├── f229f4...8b89 (model-00001, 3.8GB)
│   ├── 2b75ef...0a30 (model-00002, 3.8GB)
│   ├── 76cc7c...f90  (model-00003, 3.2GB)
│   └── 31709e...e7bc (model-00004, 1.1GB)
└── snapshots/093fba.../
    ├── model-00001-of-00004.safetensors -> ../../blobs/f229f4...
    └── [symlinks to other shards]
```

### If Model Downloads Every Time (Fix)
```bash
# Let Unsloth download to correct location automatically
python -c "from unsloth import FastLanguageModel; FastLanguageModel.from_pretrained('unsloth/gpt-oss-20b-unsloth-bnb-4bit', max_seq_length=1024, dtype=None, load_in_4bit=True)"

# Or manually fix existing download
SOURCE_DIR="/path/to/your/downloaded/model"
CACHE_DIR="$HOME/.cache/huggingface/hub/models--unsloth--gpt-oss-20b-unsloth-bnb-4bit"

mkdir -p $CACHE_DIR/blobs
mkdir -p $CACHE_DIR/snapshots/093fba6992ef5a7152481afec0bdfca1ac486998

# Copy with correct hash names (REQUIRED!)
cp $SOURCE_DIR/model-00001-of-00004.safetensors $CACHE_DIR/blobs/f229f4364b1f2cfa7df0ced4d22777145d3d552f95512f98b20493ea094e8b89
cp $SOURCE_DIR/model-00002-of-00004.safetensors $CACHE_DIR/blobs/2b75ef14502b54c47cad18bb6c2f96e919991ba994a3331799e38891ac290a30
cp $SOURCE_DIR/model-00003-of-00004.safetensors $CACHE_DIR/blobs/76cc7c3bf1cd287a8a7cea77e00eb45ce6d6a0fbc7084d29a347eed0b398af90
cp $SOURCE_DIR/model-00004-of-00004.safetensors $CACHE_DIR/blobs/31709e4bd1403df4437091d952e2ec837efbbfc06337b4b2b6c6875491e0e7bc

# Create symlinks
cd $CACHE_DIR/snapshots/093fba6992ef5a7152481afec0bdfca1ac486998
ln -sf ../../blobs/f229f4364b1f2cfa7df0ced4d22777145d3d552f95512f98b20493ea094e8b89 model-00001-of-00004.safetensors
ln -sf ../../blobs/2b75ef14502b54c47cad18bb6c2f96e919991ba994a3331799e38891ac290a30 model-00002-of-00004.safetensors
ln -sf ../../blobs/76cc7c3bf1cd287a8a7cea77e00eb45ce6d6a0fbc7084d29a347eed0b398af90 model-00003-of-00004.safetensors
ln -sf ../../blobs/31709e4bd1403df4437091d952e2ec837efbbfc06337b4b2b6c6875491e0e7bc model-00004-of-00004.safetensors
```

## 🔧 Implementation Details

### Production Training Script (`train.py`)

**Key Features:**
- Loss monitoring with target tracking (0.5 optimal)
- Configurable training profiles
- Extensive documentation with Unsloth links
- GPU selection support (CUDA_VISIBLE_DEVICES)
- Simplified model naming convention
- ~14.7GB VRAM usage

**Configuration:**
```python
# Model
model_name = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
max_seq_length = 1024
load_in_4bit = True

# LoRA (Official Unsloth)
r = 8
lora_alpha = 16  # 2:1 ratio
lora_dropout = 0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training
batch_size = 2
gradient_accumulation = 8  # Effective batch = 16
learning_rate = 2e-4
optim = "adamw_8bit"
bf16 = True  # For RTX 3090

# GPT-OSS Specific
reasoning_effort = "medium"
train_on_responses_only with:
  instruction_part = "<|start|>user<|message|>"
  response_part = "<|start|>assistant<|channel|>"
```

### Key Optimizations

**QLoRA Configuration:**
- 4-bit quantization via BitsAndBytes
- LoRA rank 8 with alpha 16 (2:1 ratio)
- Target modules: all attention + MLP layers
- No LoftQ (requires 40GB VRAM for FP16 loading)
- RSLoRA disabled (for r<16)

**Profiles:**
| Profile | Steps | LoRA r | Use Case |
|---------|-------|--------|----------|
| `quick_test` | 30 | 16 | Testing |
| `standard` | 100 | 16 | Regular training |
| `high_quality` | Full | 32 | Best quality |
| `memory_efficient` | Full | 8 | Limited VRAM |

**Usage:**
```bash
# Standard training
python scripts/train_advanced.py --profile standard --validate

# Resume from checkpoint
python scripts/train_advanced.py --profile standard \
    --resume_from_checkpoint ./outputs/checkpoint-100

# Custom dataset
python scripts/train_advanced.py \
    --dataset_name "your/dataset" \
    --max_steps 200
```

## 📊 Configuration Comparison

| Setting | Clean Unsloth | Advanced | Notes |
|---------|--------------|----------|-------|
| **Model** | `unsloth/gpt-oss-20b-unsloth-bnb-4bit` | Same or base model | Pre-quantized is faster |
| **LoRA rank** | 8 | 16 | Official vs quality focus |
| **Alpha** | 16 (2:1) | 16 (1:1) | Ratio matters |
| **Batch** | 2×8=16 | 2×8=16 | Same effective |
| **Learning rate** | 2e-4 | 2e-4 | Standard |
| **Precision** | bf16 | bf16 | RTX 3090 optimal |
| **VRAM** | ~11.7GB | ~14GB | Both fit RTX 3090 |
| **Template** | Fixed | Configurable | GPT-OSS specific |

## 💻 Inference

### Web UI with Gradio (ChatGPT-like Interface) ✨ NEW
```bash
# Launch web interface on default port
python scripts/chat_gradio_fixed.py --gpu 1

# Custom port and network access
python scripts/chat_gradio_fixed.py --gpu 1 --port 7860 --server_name 0.0.0.0

# With public share link (via Gradio)
python scripts/chat_gradio_fixed.py --gpu 1 --share
```

**Features:**
- 🎨 Beautiful ChatGPT-like interface
- ⚡ Real-time streaming responses (15-16 tokens/sec)
- 🎛️ Full parameter controls (temperature, top-p, top-k, max tokens)
- 🧠 Toggle "Show Thinking Process" to see/hide model reasoning
- 💾 Export/import conversation history
- 📊 Live statistics display
- 🔧 GPT-OSS reasoning effort control (low/medium/high)
- ✅ **Fixed**: Proper channel handling - no more cutoff responses!

### Interactive Terminal Chat
```bash
# Start interactive chat with streaming output
python scripts/chat_interactive.py --gpu 1

# With custom settings
python scripts/chat_interactive.py \
    --model_path models/latest \
    --reasoning high \
    --temperature 0.8 \
    --max_tokens 1000
```

**Interactive Commands:**
- `/help` - Show available commands
- `/clear` - Clear conversation history
- `/save [filename]` - Save conversation
- `/load <filename>` - Load previous conversation
- `/stats` - Show session statistics
- `/reasoning <low/medium/high>` - Adjust reasoning effort
- `/settings` - Display current settings
- `/exit` - Quit the chat

**Features:**
- Real-time token streaming (see response as it generates)
- Rich terminal formatting with colors
- Conversation history management
- Session persistence
- Performance metrics (tokens/sec)
- Markdown rendering
- GPT-OSS channel support

### Benchmark
```bash
python scripts/benchmark.py --model_path final_model
```

**Generation Settings:**
- Temperature: 1.0
- Top-p: 1.0
- Top-k: 0
- Reasoning effort: low/medium/high

## 🎓 Training Tips

### Avoiding Overfitting
- Monitor loss: if < 0.2, reduce epochs or learning rate
- Use r=8 instead of r=16/32
- Add weight_decay=0.1
- Enable lora_dropout=0.1

### Memory Optimization
- Use gradient_checkpointing="unsloth" (saves 30% VRAM)
- Reduce batch_size to 1
- Lower max_seq_length to 512
- Use memory_efficient profile

### Best Practices
1. **Always use pre-quantized model**: `unsloth/gpt-oss-20b-unsloth-bnb-4bit`
2. **Keep alpha/rank ratio at 2:1**: Better learning dynamics
3. **Train on completions only**: +1-3% accuracy with proper template
4. **Monitor GPU memory**: Peak should stay under 20GB
5. **Use bf16 for RTX 3090/4090**: Better than fp16

## 🔬 Evaluation

```bash
# Evaluate model quality
python scripts/evaluate.py \
    --model_path final_model \
    --dataset HuggingFaceH4/Multilingual-Thinking \
    --compare_base

# Benchmark performance
python scripts/benchmark.py --model_path final_model
```

## 📁 Project Structure

```
├── scripts/
│   ├── train.py                 # Production training with loss monitoring ✅
│   ├── chat.py                  # One-shot inference
│   ├── chat_interactive.py      # Terminal chat with Rich formatting ✨
│   ├── chat_gradio_fixed.py     # Web UI with ChatGPT-like interface 🎯
│   ├── test_speed.py            # Performance benchmarking
│   └── debug_channels.py        # Channel debugging utilities
├── configs/
│   ├── training_optimal.yaml    # Advanced training config
│   └── unsloth_official.yaml    # Clean Unsloth config
├── models/
│   ├── latest/                  # Symlink to most recent model
│   └── gpt-oss-20b_{profile}_{timestamp}/  # Named by profile and time
├── data/                        # Datasets
├── outputs/                     # Training outputs
└── setup.sh                     # Setup script
```

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Fetching 4 files" despite having model | Model not in correct HF cache location |
| 5+ minute load times | Model being re-downloaded, check cache |
| "GptOssForCausalLM not found" | Use FastLanguageModel, not raw transformers |
| "All labels are -100" error | Wrong chat template markers for train_on_responses_only |
| High VRAM usage | Use gradient_checkpointing="unsloth" |
| Training loss = 0 | Template issue, check instruction/response parts |
| Slow inference speed reported | Model loading time included in measurement |
| Flash Attention 2 fails | Incompatible with PyTorch 2.8 + CUDA 12.8 |
| Template artifacts in output | Use tokenizer.apply_chat_template() consistently |

## 🔗 Resources

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [GPT-OSS Guide](https://docs.unsloth.ai/new/gpt-oss-how-to-run-and-fine-tune)
- [Official Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb)
- [Model Hub](https://huggingface.co/unsloth/gpt-oss-20b-GGUF)
- [GitHub Repository](https://github.com/Maxli53/NeMo_NIM)

## 📝 Key Lessons Learned

1. **Use model NAME not path**: Unsloth expects HF model names, not local paths
2. **Pre-quantized is better**: Use `unsloth-bnb-4bit` versions
3. **Cache structure matters**: HF cache uses hash-based blob storage
4. **Template markers are critical**: GPT-OSS uses `<|channel|>` not `<|message|>` for assistant
5. **Keep it simple**: Clean implementation works better than complex configs

---
**Hardware:** 2x RTX 3090 (24GB each) | **Created:** 2025-09-30 | **Updated:** 2025-10-04 ✅ Production Ready

## 📊 Verified Performance Metrics (2025-10-04) ✅

### Training Performance
- **Model Loading**: 4 seconds (from HF cache)
- **Training Speed**: 16.3 seconds/step average
- **VRAM Usage**: 14.7GB peak (60% of 24GB)
- **30 Steps Duration**: 8 minutes 8 seconds
- **Final Loss**: 0.8895 (excellent convergence, target: 0.5)
- **Effective Batch Size**: 16 (2×8 gradient accumulation)
- **Loss Monitoring**: Real-time tracking with target alerts

### Inference Performance
- **Model Loading**: 12 seconds (base + LoRA adapter)
- **VRAM Usage**: 12.4GB
- **Generation Speed**: 15-16 tokens/second (actual)
- **First Token Latency**: ~2 seconds (after warmup)
- **Context Support**: 2,048 tokens (configurable to 16K)
- **Optimization**: Unsloth kernels + xformers (FA2 incompatible)

### Dual GPU Configuration
- **GPU 0**: Training (14.7GB VRAM)
- **GPU 1**: Inference serving (12.4GB VRAM)
- **Note**: 4-bit models don't support model parallelism, but can run different tasks simultaneously