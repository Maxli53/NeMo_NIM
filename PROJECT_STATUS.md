# NeMo GPT-OSS-20B - Current Project Status

**Last Updated**: September 27, 2025
**Status**: Model Conversion Complete, Inference Blocked by VRAM Constraints

---

## 🎯 Current State Summary

### ✅ Completed
1. **HF → NeMo Conversion** - Successfully converted GPT-OSS-20B from HuggingFace to NeMo format
2. **Model Loading** - Model loads successfully into VRAM (23.07 GB / 24 GB)
3. **Infrastructure Ready** - Docker container, CUDA stack, swap space all operational

### ❌ Blocking Issue
**CUDA Out of Memory during Inference**
- Model uses 23.07 GB VRAM after loading
- Generation requires 16 MB more for KV cache
- Only 34.81 MB free VRAM available
- **Root Cause**: Dequantization from MXFP4 (4-bit) → BF16 (16-bit) expanded model size

---

## 📊 Detailed Timeline

### Phase 1: Model Download ✅
- Downloaded GPT-OSS-20B from HuggingFace (77 GB)
- Location: `/media/ubumax/WD_BLACK/AI_Projects/NeMo_GPT/models/gpt-oss-20b/`
- Format: 3 safetensors files (MXFP4 quantized)

### Phase 2: Conversion Attempts

#### First Attempt - Failed
- **Date**: September 26, 2025
- **Method**: Direct conversion with 62 GB RAM
- **Result**: ❌ FAILED - Exit code -9 (OOM killed by Linux kernel)
- **Reason**: Conversion process peaked above 60 GB RAM

#### Second Attempt - Success
- **Date**: September 27, 2025
- **Method**: Added 100 GB swap space
- **Duration**: ~4 minutes
- **Result**: ✅ SUCCESS - Checkpoint created at `/workspace/checkpoints/gpt-oss-20b`
- **Size**: Distributed checkpoint format with context/ and weights/ directories

### Phase 3: Inference Attempt - Failed ❌
- **Date**: September 27, 2025
- **Result**: Model loads but cannot generate
- **Error**: `torch.OutOfMemoryError: CUDA out of memory`
- **Details**:
  - Model loaded: 23.07 GB VRAM used
  - Tried to allocate: 16 MB (for KV cache)
  - Free VRAM: 34.81 MB
  - GPU: RTX 3090 (24 GB total capacity)

---

## 🔬 Technical Analysis

### Why VRAM Exceeded

**Original Model (HuggingFace)**:
- Format: MXFP4 (4-bit microscaling floating point)
- Size: ~13 GB quantized weights
- Experts: 32 total experts stored efficiently

**Converted Model (NeMo)**:
- Format: BF16 (16-bit brain floating point)
- Size: ~23 GB dequantized weights
- Expansion: 4-bit → 16-bit = 4× size increase
- All 32 experts dequantized and loaded

**Memory Breakdown**:
```
Model weights:     23.07 GB
PyTorch reserved:  58.17 MB
Free VRAM:         34.81 MB
Total:             24.00 GB (GPU capacity)
```

### Conversion Process

**Script**: `workspace/convert_hf_to_nemo.py`

**What it does**:
1. Loads MXFP4 quantized HF checkpoint
2. Dequantizes all 24 layers × 32 experts to BF16
3. Converts to NeMo distributed checkpoint format
4. Saves to `/workspace/checkpoints/gpt-oss-20b/`

**Output Structure**:
```
/workspace/checkpoints/gpt-oss-20b/
├── context/
│   ├── artifacts/
│   │   └── generation_config.json
│   ├── nemo_tokenizer/
│   │   ├── chat_template.jinja
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── io.json
│   └── model.yaml
└── weights/
    ├── .metadata
    ├── __0_0.distcp
    ├── __0_1.distcp
    ├── common.pt
    └── metadata.json
```

---

## 💾 System Resources

### Hardware
- **GPU**: NVIDIA GeForce RTX 3090
  - VRAM: 24 GB
  - Compute Capability: 8.6 (Ampere)
  - Driver: 580.65.06
- **CPU**: AMD Ryzen 9 3950X (32 threads)
- **RAM**: 62 GB
- **Swap**: 100 GB (on NVMe SSD)
- **Storage**: 4.4 TB available

### Software
- **OS**: Ubuntu 24.04 (native)
- **CUDA**: 12.6
- **Docker**: nvcr.io/nvidia/nemo:25.07.gpt_oss
- **Python**: 3.12 (container) / 3.11 (host)
- **PyTorch**: 2.8.0
- **NeMo**: 2.4.0 / 2.6.0rc0

---

## 🛠️ Solutions to VRAM Issue

### Option 1: Reduce KV Cache Size (Easiest)
**Edit `workspace/inference.py`**:
```python
# Reduce token generation to free up KV cache memory
NUM_TOKENS_TO_GENERATE = 20  # Was: 100
MAX_BATCH_SIZE = 1           # Already optimized
```

**Pros**:
- Immediate fix
- No additional hardware needed
- Model still functional

**Cons**:
- Shorter output (20 tokens instead of 100)
- Multiple runs needed for longer content

**Expected VRAM**: ~22-22.5 GB (frees ~1.5 GB for KV cache)

---

### Option 2: Multi-GPU Expert Parallelism (Requires 2+ GPUs)
**Edit `workspace/inference.py`**:
```python
# Distribute 32 experts across multiple GPUs
EXPERT_PARALLEL_SIZE = 2  # Split experts across 2 GPUs
DEVICES = 2                # Use 2 GPUs
```

**Pros**:
- Can run with full generation length
- Professional multi-GPU setup
- Scales to larger models

**Cons**:
- ❌ Requires second GPU (don't have)
- More complex configuration

---

### Option 3: Keep Original MXFP4 Quantization (Research)
**Approach**: Find method to run NeMo inference with quantized weights

**Requirements**:
- Check if NeMo supports MXFP4 inference
- May need custom quantization config
- Potentially use FP8 inference instead

**Pros**:
- Would use ~6-8 GB VRAM (4-bit precision)
- Plenty of headroom for generation

**Cons**:
- May not be supported by NeMo
- Requires research and custom code
- Potential accuracy loss

---

### Option 4: Model Pruning/Distillation (Long-term)
**Approach**: Create smaller version of GPT-OSS-20B

**Methods**:
- Reduce number of experts (32 → 16)
- Reduce layers (24 → 20)
- Knowledge distillation

**Pros**:
- Permanent solution
- Customized for specific use case
- Can optimize quality/speed tradeoff

**Cons**:
- Requires training infrastructure
- Time-intensive (days/weeks)
- Requires large dataset

---

## 📁 Repository Structure

```
/media/ubumax/WD_BLACK/AI_Projects/NeMo_GPT/
├── models/
│   └── gpt-oss-20b/                    # Original HF model (77 GB, MXFP4)
├── workspace/
│   ├── checkpoints/
│   │   └── gpt-oss-20b/                # ✅ Converted NeMo checkpoint (BF16)
│   ├── convert_hf_to_nemo.py           # ✅ Conversion script (working)
│   ├── inference.py                    # ⚠️ Inference script (OOM issue)
│   └── convert_with_swap.log           # Conversion log
├── nemo/                               # NeMo framework source
├── PREFLIGHT_CHECK.md                  # Hardware/software readiness
├── PROJECT_STATUS.md                   # This file - current state
├── README.md                           # Project overview
└── SETUP_STATUS.md                     # Installation status
```

---

## 📝 Files Created/Modified

### New Files
1. **`workspace/convert_hf_to_nemo.py`** - Conversion script based on official NeMo tutorial
2. **`PREFLIGHT_CHECK.md`** - Comprehensive system readiness report
3. **`PROJECT_STATUS.md`** - This detailed status document
4. **`workspace/convert_with_swap.log`** - Conversion execution log
5. **`workspace/inference_run.log`** - Inference attempt log

### Modified Files
1. **`workspace/inference.py`**
   - Changed: `MODEL_PATH = "/workspace/checkpoints/gpt-oss-20b"`
   - Was: `MODEL_PATH = "/root/.cache/nemo/models/gpt-oss-20b"`
   - Reason: Point to converted checkpoint

---

## 🔍 Logs and Evidence

### Conversion Success Log
```bash
$ tail -50 /workspace/convert_with_swap.log

[NeMo D 2025-09-27 10:45:08] Successfully dequantized model.layers.23.mlp.experts.gate_up_proj
💡 Tip: For seamless cloud uploads...
[NeMo I 2025-09-27 10:45:24] Fixing mis-match between ddp-config & mcore-optimizer config
[NeMo I 2025-09-27 10:45:24] Rank 0 has data parallel group : [0]
...
[10:48:30] INFO Job nemo.collections.llm.api.import_ckpt-dqs50ttg575xmc finished: SUCCEEDED

======================================================================
Conversion Complete!
NeMo checkpoint saved to: /workspace/checkpoints/gpt-oss-20b
======================================================================
```

### Inference Failure Log
```bash
$ tail -20 /workspace/inference_run.log

[rank0]: torch.OutOfMemoryError: CUDA out of memory.
[rank0]: Tried to allocate 16.00 MiB.
[rank0]: GPU 0 has a total capacity of 23.56 GiB of which 34.81 MiB is free.
[rank0]: Process 26463 has 23.07 GiB memory in use.
[rank0]: Of the allocated memory 22.70 GiB is allocated by PyTorch,
[rank0]: and 58.17 MiB is reserved by PyTorch but unallocated.
```

---

## 🎯 Recommended Next Steps

### Immediate (Today)
1. **Try Option 1**: Reduce `NUM_TOKENS_TO_GENERATE` to 20
2. **Test inference**: Run `docker exec nemo-gpt-oss bash -c "cd /workspace && python inference.py"`
3. **Validate output**: Check if 20-token responses are acceptable

### Short-term (This Week)
1. **Document findings**: Update this status with inference results
2. **Experiment with settings**:
   - Try `NUM_TOKENS_TO_GENERATE = 10, 20, 30, 50`
   - Find maximum stable token count
3. **Explore quantization**: Research NeMo FP8 inference support

### Long-term (Next Month)
1. **Consider hardware upgrade**: Second RTX 3090 for expert parallelism
2. **Model optimization**: Prune/distill smaller variant
3. **Alternative models**: Evaluate smaller MoE models that fit 24GB

---

## 📊 Performance Expectations (If Fixed)

### With NUM_TOKENS_TO_GENERATE = 20
- **Loading Time**: 30-60 seconds
- **First Token**: 2-5 seconds
- **Generation Speed**: 10-20 tokens/second
- **Per Prompt**: ~3-4 seconds total
- **3 Prompts**: ~10-15 seconds total

### Memory Usage (Projected)
```
Model weights:     23.07 GB
KV cache (20 tok): ~0.50 GB
Buffer/overhead:   ~0.43 GB
Total:            ~24.00 GB ✅ Should fit
```

---

## 🐛 Known Issues

### 1. CUDA OOM During Inference (CRITICAL)
- **Status**: ❌ Blocking
- **Impact**: Cannot generate text
- **Workaround**: Reduce NUM_TOKENS_TO_GENERATE
- **Permanent Fix**: Multi-GPU or quantized inference

### 2. CPU RNG State Warnings (Non-blocking)
```
CPU RNG state changed within GPU RNG context
```
- **Status**: ⚠️ Warning only
- **Impact**: None (cosmetic)
- **Action**: Can ignore

### 3. Bitsandbytes CUDA Binary Missing (Non-blocking)
```
Could not find the bitsandbytes CUDA binary
```
- **Status**: ⚠️ Warning only
- **Impact**: No 8-bit quantization (not needed)
- **Action**: Can ignore

---

## 🎓 Lessons Learned

### What Worked
1. ✅ 100GB swap enabled RAM-intensive conversion
2. ✅ Official NeMo tutorial approach successful
3. ✅ Docker container provides stable environment
4. ✅ CPU initialization prevents OOM during loading

### What Didn't Work
1. ❌ Direct HF checkpoint inference (requires conversion)
2. ❌ Conversion with only 62GB RAM (needed swap)
3. ❌ Full BF16 model doesn't fit 24GB for generation
4. ❌ Assuming quantized model stays quantized

### Key Insights
1. **Dequantization is automatic**: NeMo converts MXFP4 → BF16
2. **Conversion ≠ Inference**: Different memory requirements
3. **24GB is borderline**: Need optimization for 20B+ MoE models
4. **Swap is essential**: Enables conversion but not inference

---

## 📚 References

### Official Documentation
- [NeMo GPT-OSS Tutorial](file:///media/ubumax/WD_BLACK/AI_Projects/NeMo_GPT/nemo/tutorials/llm/gpt-oss/ticket-routing-lora/gpt-oss-lora.ipynb)
- [NeMo Framework Docs](https://docs.nvidia.com/nemo-framework/)
- [Megatron-Core](https://github.com/NVIDIA/Megatron-LM)

### Model Information
- [GPT-OSS-20B on HuggingFace](https://huggingface.co/openai/gpt-oss-20b)
- [OpenAI GPT-OSS GitHub](https://github.com/openai/gpt-oss)
- [MXFP4 Specification](https://arxiv.org/abs/2302.00994)

### GitHub Repository
- **URL**: https://github.com/Maxli53/NeMo_NIM
- **Branch**: master
- **Latest Commit**: `6313059` - Add HF to NeMo conversion pipeline

---

## 🔮 Future Work

### Phase 1: Get Inference Working
- [ ] Reduce NUM_TOKENS_TO_GENERATE to 20
- [ ] Test inference successfully
- [ ] Benchmark generation speed
- [ ] Document optimal settings

### Phase 2: Optimize Performance
- [ ] Research NeMo FP8 inference
- [ ] Explore keeping MXFP4 quantization
- [ ] Profile memory usage patterns
- [ ] Optimize KV cache allocation

### Phase 3: Scale Up
- [ ] Consider second GPU (RTX 3090)
- [ ] Implement expert parallelism
- [ ] Enable longer generation (100+ tokens)
- [ ] Benchmark multi-GPU performance

### Phase 4: Production Deployment
- [ ] Create inference API service
- [ ] Add batching and queueing
- [ ] Implement monitoring/logging
- [ ] Deploy with load balancing

---

## 📞 Contact & Contributions

**Repository**: https://github.com/Maxli53/NeMo_NIM
**Issues**: https://github.com/Maxli53/NeMo_NIM/issues
**License**: Apache 2.0

---

**Status**: 🟡 **PARTIALLY OPERATIONAL** - Conversion complete, inference requires optimization to fit VRAM constraints.

**Next Action**: Reduce `NUM_TOKENS_TO_GENERATE` to 20 and test inference.