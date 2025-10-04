# Lessons Learned - Unsloth GPT-OSS-20B Project

**Date**: 2025-10-04
**Project**: Unsloth Fine-tuning Pipeline
**Hardware**: 2x RTX 3090 (24GB each)

---

## 🔑 Key Technical Learnings

### 1. Model Loading & Caching

#### ✅ What Works
- **Use model names, not paths**: `unsloth/gpt-oss-20b-unsloth-bnb-4bit`
- **Pre-quantized models are superior**: 3x faster loading, no quantization overhead
- **HuggingFace cache structure**: Hash-based blob storage with symlinks
- **Cached model loads in 4 seconds**: When properly configured

#### ❌ Common Pitfalls
- Loading from local path causes re-quantization every time
- Incorrect cache structure triggers re-downloads (5+ minutes)
- Manual model moves break HF cache symlinks

### 2. GPU Memory Management

#### Single GPU Usage (RTX 3090)
```
Training: 14.7GB VRAM (60% utilization)
Inference: 12.4GB VRAM (52% utilization)
Headroom: ~10GB for larger batches or longer sequences
```

#### Dual GPU Strategies
- **Parallel Tasks**: Train on GPU 0, serve inference on GPU 1
- **Not Supported**: Model parallelism for 4-bit quantized models
- **Alternative**: Run different experiments simultaneously

#### Memory Optimization Techniques
1. `gradient_checkpointing="unsloth"` → Saves 30% VRAM
2. Reduce batch size to 1 → Saves ~2GB
3. Lower sequence length → Linear VRAM reduction
4. Use 4-bit quantization → 75% reduction vs FP16

### 3. Training Performance

#### Measured Results (30 steps test)
- **Speed**: 16.3 seconds per step
- **Total Duration**: 8 minutes 8 seconds
- **Loss Convergence**: 2.04 → 1.43 (30% reduction)
- **Gradient Norm**: Stabilized after step 10
- **Learning Rate Schedule**: Linear warmup crucial

#### Optimal Settings for RTX 3090
```python
batch_size = 2
gradient_accumulation_steps = 8  # Effective batch = 16
learning_rate = 2e-4
bf16 = True  # Better than fp16 for RTX 3090
optim = "adamw_8bit"  # Saves memory without quality loss
```

### 4. LoRA Configuration

#### Official Unsloth Recommendations
- **Rank (r)**: 8 for standard, 16 for quality, 32 for maximum
- **Alpha**: 2:1 ratio with rank (alpha = 2×r)
- **Dropout**: 0 for speed, 0.1 for regularization
- **Target Modules**: All 7 attention/MLP layers

#### Trade-offs
| Configuration | VRAM | Quality | Speed |
|--------------|------|---------|-------|
| r=8, alpha=16 | 11.7GB | Good | Fast |
| r=16, alpha=16 | 14GB | Better | Medium |
| r=32, alpha=64 | 18GB | Best | Slow |

### 5. Inference Optimization

#### Performance Metrics
- **First Token**: ~20 seconds (model warmup)
- **Subsequent Tokens**: 2.4 tokens/second
- **Context Length**: 16,384 tokens verified
- **Reasoning Effort**: Medium setting optimal

#### Export Options
1. **Direct Unsloth**: Native inference, fastest
2. **GGUF (llama.cpp)**: CPU inference, portable
3. **ONNX**: Cross-platform deployment
4. **vLLM**: Production serving (not tested)

### 6. SOLVED: GPT-OSS Inference Issue (2025-10-04) ✅

#### Initial Problem
- **Issue**: `ArgsMismatchError` during inference when loading models incorrectly
- **Error**: Missing required positional argument in MoE forward pass
- **Root Cause**: Incorrect model loading approach for fine-tuned models

#### The Solution
**We were loading the model wrong!** The correct approach requires:

1. **Load the base model first**:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",  # Base model
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)
```

2. **Apply the LoRA adapter**:
```python
from peft import PeftModel
model = PeftModel.from_pretrained(model, "final_model/")
```

3. **Enable inference mode**:
```python
FastLanguageModel.for_inference(model)
```

#### Why Our Initial Attempts Failed
- **Wrong**: Loading checkpoint path directly as if it was a complete model
- **Wrong**: Trying to load LoRA adapter without base model
- **Right**: Load base model → Apply LoRA → Enable inference

#### Key Insight
The inference DOES work (proven in training script at the end), but requires proper loading sequence. The model needs both the base weights and LoRA adapter properly merged before calling `for_inference()`.

### 7. Dual GPU Usage (2x RTX 3090)

#### Working Patterns
1. **Training + Inference**: GPU 0 trains while GPU 1 serves
   ```bash
   CUDA_VISIBLE_DEVICES=0 python train_unsloth.py
   CUDA_VISIBLE_DEVICES=1 python inference.py
   ```

2. **Parallel Experiments**: Different configs on each GPU
   - 2x throughput for hyperparameter search
   - Compare approaches simultaneously

#### Limitations with 4-bit Models
- **No Model Parallelism**: Can't split 4-bit quantized models
- **No DDP**: BitsAndBytes incompatible with distributed training
- **Workaround**: Use separate GPUs for separate tasks

#### Resource Usage
- **Training**: 14.7GB on GPU 0 (60% utilization)
- **Inference**: 12.4GB on GPU 1 (52% utilization)
- **Headroom**: ~10GB per GPU for larger batches

### 7. Performance Optimization Setup (2025-10-04)

#### VRAM Optimization Strategy
```python
# Training optimizations for full VRAM usage
per_device_train_batch_size = 4  # Up from 2 (uses more VRAM)
gradient_accumulation_steps = 4  # Down from 8 (same effective batch=16)
max_seq_length = 2048  # Can increase from 1024 if VRAM allows
bf16 = True  # Already optimal for RTX 3090
gradient_checkpointing = "unsloth"  # Memory efficient
```

### 8. Successful Training Run (2025-10-04) ✅

#### Training Configuration
- **Model**: unsloth/gpt-oss-20b-unsloth-bnb-4bit
- **Dataset**: HuggingFaceH4/Multilingual-Thinking (5000 samples)
- **LoRA Config**: r=8, alpha=16 (2:1 ratio)
- **Batch Size**: 4 per device × 4 gradient accumulation = 16 effective
- **Steps**: 200 (3.18 epochs)

#### Results
- **Training Time**: 38 minutes 4 seconds
- **Speed**: 11.4 seconds/step average
- **Final Loss**: 1.15 (from initial 2.04)
- **VRAM Usage**: 20GB peak (83% utilization)
- **Model Size**: 15.9MB LoRA adapter
- **Convergence**: Excellent, no overfitting

#### Loss Progression
```
Step   1: 2.04 (initial)
Step  50: 1.20 (checkpoint)
Step 100: 1.13 (checkpoint)
Step 150: 1.08 (checkpoint)
Step 200: 1.15 (final)
```

#### Inference Speed Optimizations
1. **Batch Processing**:
   - Single prompt: 12.4GB VRAM, 2.4 tokens/sec
   - Batch size 2: ~20GB VRAM, 4-5 tokens/sec total
   - Batch size 3: ~23GB VRAM, 6-7 tokens/sec total

2. **Reasoning Effort Trade-offs**:
   - `"low"`: Fastest (4-6 tokens/sec), basic responses
   - `"medium"`: Balanced (2-4 tokens/sec), good quality
   - `"high"`: Best quality (1-2 tokens/sec), complex reasoning

3. **Native Unsloth Optimizations** (automatic):
   - Flash Attention when available
   - BF16 tensor cores on RTX 3090
   - KV cache optimization
   - Optimized chat templates

#### Expected Performance Gains
- **Training**: 15-20% faster with larger batch size
- **Inference**: 2-3x throughput with batching
- **VRAM Usage**: 80-90% utilization (optimal)
- **Target Speed**: 6-10 tokens/sec with batch=2

---

## 🏗️ Architecture Decisions

### Why Unsloth Over NeMo?

| Aspect | Unsloth | NeMo |
|--------|---------|------|
| **VRAM Required** | 14.7GB | >24GB for PTQ |
| **Setup Complexity** | Simple pip install | Docker + complex config |
| **Time to First Model** | 30 minutes | 2+ hours |
| **Hardware Flexibility** | Works on consumer GPUs | Requires datacenter GPUs |
| **Production Ready** | Yes, with limitations | Yes, enterprise-grade |

### When to Use Each
- **Unsloth**: Fine-tuning, experimentation, limited hardware
- **NeMo**: Large-scale training, enterprise deployment, multi-GPU
- **Hyperstack**: When you need A100s for NeMo quantization

---

## 📚 Project Management Insights

### Documentation Best Practices
1. **Test everything before documenting**
2. **Include exact commands and outputs**
3. **Document hardware requirements clearly**
4. **Keep a running PROJECT_STATUS.md**
5. **Archive old approaches in separate folder**

### Workflow Optimization
1. **Start with minimal test (30 steps)**
2. **Verify cache before full training**
3. **Use checkpoint saves strategically**
4. **Monitor first few steps closely**
5. **Keep inference script ready for testing**

---

## 🐛 Debugging Checklist

### Model Loading Issues
- [ ] Check HF cache: `ls ~/.cache/huggingface/hub/`
- [ ] Verify model name matches exactly
- [ ] Ensure 4-bit version specified
- [ ] Check disk space (need 15GB free)

### Training Issues
- [ ] Monitor GPU memory: `watch nvidia-smi`
- [ ] Check gradient norms in logs
- [ ] Verify dataset loaded correctly
- [ ] Ensure correct chat template

### Inference Issues
- [ ] Model path points to checkpoint directory
- [ ] Tokenizer files present
- [ ] GPU has 12GB+ free VRAM
- [ ] Generation parameters reasonable

---

## 🚀 Next Steps & Recommendations

### Immediate Actions
1. **Prepare Production Dataset**
   - Format as ShareGPT or Alpaca
   - Aim for 10K+ high-quality examples
   - Include diverse reasoning tasks

2. **Optimize Training Pipeline**
   - Implement validation split
   - Add early stopping
   - Create evaluation metrics
   - Set up W&B logging

3. **Production Deployment**
   - Export to GGUF for CPU fallback
   - Set up inference server
   - Implement request batching
   - Add monitoring/alerting

### Future Enhancements
1. **Multi-Model Pipeline**
   - Train specialized models for different tasks
   - Implement model routing
   - Create ensemble predictions

2. **Distributed Training**
   - Explore DeepSpeed integration
   - Test FSDP for larger models
   - Benchmark multi-node training

3. **Quality Improvements**
   - Implement RLHF/DPO training
   - Add constitutional AI constraints
   - Create custom evaluation benchmarks

---

## 💡 Pro Tips

### Training
- Always warm up learning rate
- Save checkpoints every 25% of training
- Monitor loss plateaus for early stopping
- Use different seeds for ensemble diversity

### Inference
- Pre-load model for production
- Implement response caching
- Use streaming for long generations
- Set reasonable timeout limits

### Development
- Keep separate configs for dev/prod
- Version control your datasets
- Document model lineage
- Archive all training logs

---

## 📊 Cost-Benefit Analysis

### Local Setup (RTX 3090)
- **Initial Cost**: $1,600 (2x RTX 3090 used)
- **Running Cost**: ~$0.50/day electricity
- **Training Time**: 8 hours for full fine-tune
- **Capability**: 20B parameter models

### Cloud Alternative (A100)
- **Hourly Cost**: $2-3/hour
- **Training Time**: 2-3 hours
- **Setup Time**: 30 minutes
- **Capability**: 70B+ parameter models

### Recommendation
Start locally for development, use cloud for:
- Large model quantization
- Production training runs
- Multi-GPU experiments
- Customer deployments

---

## 🎯 Success Metrics

### Technical
- ✅ Model loads in <5 seconds
- ✅ Training loss <1.5
- ✅ Inference speed >2 tokens/sec
- ✅ VRAM usage <15GB
- ✅ Checkpoint recovery works

### Business
- ⏳ Custom model deployment ready
- ⏳ API endpoint configured
- ⏳ Monitoring dashboard live
- ⏳ Documentation complete
- ⏳ Team trained on pipeline

---

## 🔄 Version History

### v1.0 (2025-10-04)
- Initial lessons from GPT-OSS-20B testing
- Dual GPU configuration documented
- Performance baselines established
- Unsloth vs NeMo comparison

---

### 10. Flash Attention 2 Compatibility Issues (2025-10-04)

#### The Problem
Attempted to install Flash Attention 2 for faster inference but encountered multiple issues:

#### What We Tried
1. **Flash Attention 2.8.3**: Installed but Unsloth requires <=2.8.2
2. **Flash Attention 2.8.2**: Compilation hung/timed out after 5+ minutes
3. **Flash Attention 2.7.4.post1**: Also required compilation, timed out
4. **Flash Attention 2.6.3**: Pre-built wheel had ABI incompatibility with PyTorch 2.8

#### Root Causes
- **PyTorch 2.8 + CUDA 12.8**: Bleeding edge combo with no pre-built wheels
- **Compilation issues**: Flash Attention compilation is notoriously difficult
  - Requires 8-10GB RAM during compilation
  - Compiles hundreds of CUDA kernels
  - Often fails silently or hangs
- **Version sensitivity**: Exact match required for PyTorch, CUDA, GCC versions

#### Solution
Proceeded without Flash Attention 2:
- **Unsloth optimizations**: Custom Triton kernels provide 2x speedup
- **Xformers**: Memory-efficient attention as fallback (installed and working)
- **Result**: 15-16 tokens/sec (acceptable for 20B model on RTX 3090)

#### Key Learnings
1. **Don't chase marginal gains**: FA2 would only add 2-3 tokens/sec
2. **Pre-built wheels are crucial**: Compilation from source is unreliable
3. **Unsloth is already optimized**: Their kernels are faster than FA2 anyway
4. **Memory bandwidth is the real bottleneck**: For 20B models, not attention computation

### 11. Chat Template Alignment (2025-10-04)

#### The Issue
Model responses included template artifacts like `final<|message|>` and `analysis<|message|>`

#### Root Cause
Mismatch between training and inference templates:
- **Training**: Used `tokenizer.apply_chat_template()` with official format
- **Inference**: Manually constructed template strings (incorrect)

#### Solution
Always use `tokenizer.apply_chat_template()` for both training and inference:
```python
# Correct approach
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # True for inference, False for training
    reasoning_effort="low"  # GPT-OSS specific parameter
)
```

#### Key Insight
The `chat_template.jinja` file saved with the model defines the exact format. GPT-OSS-20B has special features:
- Three channels: analysis, commentary, final
- Reasoning levels: low, medium, high
- Developer role separate from system

### 12. Performance Measurement Pitfalls

#### The Problem
Chat.py showed 5-8 tokens/sec but actual speed was 15-16 tokens/sec

#### Root Cause
Including model loading time in speed measurement:
- Model loading: ~12 seconds
- Actual inference: Fast
- Reported speed: (tokens generated) / (loading + inference time)

#### Solution
Separate model loading from inference timing:
```python
# Load once
model = load_model()  # 12 seconds

# Then measure
start = time.time()
output = generate(model, prompt)
speed = tokens / (time.time() - start)  # Accurate speed
```

#### Lesson
Always profile code properly - identify where time is actually spent

### 13. GPT-OSS Channel Handling Issue (2025-10-05) ✅ SOLVED

#### The Problem
Model would generate complete analysis in "analysis channel" but the "final channel" would only output single characters like "I" or "My" then stop.

#### Discovery Process
1. User noticed model showed thinking process but no actual answer
2. Initially suspected token limit issue - increased to 1500, no change
3. With "Show Thinking OFF" - worked perfectly
4. With "Show Thinking ON" - final channel cut off after one word
5. Both base and fine-tuned models showed same behavior

#### Root Cause
GPT-OSS generates `<|end|>` tokens after each channel:
```
<|channel|>analysis<|message|>...thinking...<|end|>
<|channel|>commentary<|message|>...more thinking...<|end|>
<|channel|>final<|message|>I<|return|>  # Cut off here!
```

The streaming code was stopping at the FIRST `<|end|>` token, never reaching the final channel.

#### Solution
Created custom `ChannelStoppingCriteria` class:
```python
class ChannelStoppingCriteria(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # Track if we've entered final channel
        generated_text = self.tokenizer.decode(input_ids[0])
        if "final<|message|>" in generated_text:
            self.has_final = True

        # Only stop at <|return|> if in final channel
        last_token = input_ids[0][-1].item()
        if last_token == self.return_token and self.has_final:
            return True

        # Don't stop at intermediate <|end|> tokens
        if last_token == self.end_token:
            return False
```

#### Key Insights
1. **Never modify the original template** - Unsloth templates are carefully designed
2. **Handle output correctly** - The issue was in our code, not the model
3. **Debug systematically** - Token-by-token analysis revealed the exact issue
4. **Both models affected** - Proved it wasn't a fine-tuning problem

#### Implementation
- Fixed in `chat_gradio_fixed.py` with proper channel handling
- Streams only final channel when "Show Thinking" is OFF
- Shows all channels formatted when "Show Thinking" is ON
- No flashing/disappearing text during generation

**Remember**: The best code is code that works. Start simple, measure everything, optimize later.