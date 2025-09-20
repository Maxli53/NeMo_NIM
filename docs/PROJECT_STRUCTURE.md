# Project Structure

## Overview

This document describes the organized structure of the AI Agents GPT-OSS-20B MoE project.

## Directory Structure

```
AI_agents/
├── configs/                   # Configuration files
│   └── *.yaml                 # Various config files
│
├── data/                      # Data files and knowledge bases
│   └── *.pdf                  # PDF documents for agent knowledge
│
├── docs/                      # All documentation (consolidated)
│   ├── 00_DOCUMENTATION_INDEX.md         # Master documentation index
│   ├── 01_PROJECT_OVERVIEW.md            # Project overview
│   ├── 02_QUICK_START.md                 # Quick start guide
│   ├── 11_NATIVE_MOE_IMPLEMENTATION.md   # MoE implementation details
│   ├── 12_OPTIMIZATIONS_COMPLETE.md      # Completed optimizations
│   ├── 13_OPTIMIZATION_ROADMAP.md        # Future optimizations
│   ├── 40_MULTI_AGENT_OVERVIEW.md        # Multi-agent system overview
│   ├── 41_AGENT_INTEGRATION.md           # Agent-MoE integration
│   ├── API_DOCUMENTATION.md              # API documentation
│   ├── ARCHITECTURE.md                   # System architecture
│   ├── CLAUDE.md                         # Project instructions
│   ├── COMPLETE_TEST_REPORT.md           # Complete test report
│   ├── CONFIGURATION.md                  # Configuration guide
│   ├── DEPLOYMENT.md                     # Deployment guide
│   ├── GPT_OSS_MODEL.md                  # GPT-OSS model details
│   ├── OPTIMIZATION_GUIDE.md             # Optimization guide
│   ├── PROJECT_STRUCTURE.md              # This document
│   └── TROUBLESHOOTING.md                # Troubleshooting guide
│
├── gpt-oss-20b/              # GPT-OSS-20B model files
│   ├── experts/              # Expert weights
│   └── routers/              # Router weights
│
├── logs/                     # Application logs
│
├── scripts/                  # Utility scripts
│   ├── download_models.py   # Model download script
│   └── ...
│
├── src/                      # Source code
│   ├── agents/               # Agent implementations
│   │   ├── expert.py         # Expert agent
│   │   ├── consensus.py     # Consensus agent
│   │   └── integrated_multi_agent_gptoss.py
│   ├── core/                 # Core functionality
│   │   ├── moderator.py     # Discussion moderator
│   │   ├── session.py       # Session management
│   │   └── vector_db.py     # FAISS vector database
│   ├── moe/                  # MoE implementation
│   │   ├── async_expert_loader.py     # Async I/O prefetching
│   │   ├── cuda_kernels.py            # CUDA kernel fusion
│   │   ├── expert_cache.py            # Expert caching
│   │   ├── expert_mixer.py            # Expert mixing
│   │   ├── moe_config.py              # MoE configuration
│   │   ├── multi_gpu_moe.py           # Multi-GPU support
│   │   ├── native_moe_complete.py     # Complete MoE implementation
│   │   └── tiered_cache.py            # Tiered caching system
│   ├── ui/                   # UI components
│   │   └── streamlit_app.py # Streamlit dashboard
│   └── utils/                # Utility functions
│
├── test_results/             # Test output files
│   ├── moe_checklist_results.json
│   ├── moe_full_edge_perf_results.json
│   ├── optimized_metrics.json
│   └── test_results_v3.json
│
├── tests/                    # Test files
│   ├── benchmark_native_moe.py        # Performance benchmarks
│   ├── model_integration_test.py      # Integration tests
│   ├── test_moe_checklist.py          # MoE validation tests
│   ├── test_optimizations.py          # Optimization tests
│   └── test_suite_v3.py               # Complete test suite
│
├── venv_gptoss/              # Virtual environment
│
├── README.md                 # Main readme
├── config.yaml               # Main configuration
├── main.py                   # Entry point
├── pyproject.toml            # Python project config
└── requirements.txt          # Dependencies
```

## Key File Locations

### Documentation
- **All Documentation**: `/docs/` - All MD files consolidated in one location
- **Numbered Docs**: Following logical order (00-49 system)
- **No duplicates**: All documentation in single location

### Source Code
- **MoE Implementation**: `/src/moe/` - All MoE-related code
- **Multi-Agent System**: `/src/agents/` - Agent implementations
- **Core Logic**: `/src/core/` - Core system components

### Tests & Results
- **Test Scripts**: `/tests/` - All test files
- **Test Results**: `/test_results/` - JSON output from tests

### Models & Data
- **Model Weights**: `/gpt-oss-20b/` - GPT-OSS-20B model files
- **Knowledge Bases**: `/data/` - PDF files for agent knowledge

## Important Documentation Files

All important documentation is now consolidated in `/docs/`:
1. **COMPLETE_TEST_REPORT.md** - Comprehensive test results with all validation
2. **OPTIMIZATION_GUIDE.md** - Detailed optimization implementation guide
3. **CLAUDE.md** - Project instructions and context
4. **13_OPTIMIZATION_ROADMAP.md** - Next phase implementation plan

## Documentation Numbering System

- **00-09**: Index and overview documents
- **10-19**: Implementation details
- **20-29**: Guides and tutorials
- **30-39**: Configuration and deployment
- **40-49**: Multi-agent system
- **50+**: Additional topics

## Next Steps

With the project now properly organized, the next step is to proceed with the optimizations outlined in `/docs/13_OPTIMIZATION_ROADMAP.md`:

1. Phase 1: Quick Wins (Dynamic Batching, Flash Attention)
2. Phase 2: Quantization Pipeline (INT8 weights)
3. Phase 3: Advanced Kernel Optimizations
4. Phase 4: Experimental Optimizations
5. Phase 5: Production Hardening

---

*Last Updated: September 2025*
*Organization Complete: All files preserved and properly structured*