# CLAUDE.md - Development Guidelines & Safety Protocols

## 🎯 Purpose
This document ensures we maintain professional development standards and never compromise on quality, safety, or best practices for the integrated AI Agents + MoE system.

## 🛡️ Core Principles

### 1. Safety First
- **NEVER** deploy untested code to production
- **NEVER** enable experimental features without thorough testing
- **ALWAYS** have rollback mechanisms ready
- **ALWAYS** test on staging before production

### 2. No Shortcuts
- **NO** hardcoded values (use configs/environment variables)
- **NO** commented-out code in production
- **NO** print statements for debugging (use proper logging)
- **NO** direct database queries (use ORM/prepared statements)
- **NO** secrets in code (use .env or secret management)

### 3. Transparency
- **ALWAYS** document what actually works vs what's planned
- **NEVER** claim performance without real benchmarks
- **ALWAYS** be honest about limitations and issues

## 📋 Development Checklist

### Before Starting Any Task
- [ ] Read existing documentation
- [ ] Check if similar code exists (avoid duplication)
- [ ] Verify dependencies are already in project
- [ ] Create feature branch from main/master
- [ ] Update TODO list for tracking

### During Development
- [ ] Follow existing code patterns and style
- [ ] Add type hints to all functions
- [ ] Write docstrings for complex functions
- [ ] Use meaningful variable names
- [ ] Keep functions small and focused (<50 lines)
- [ ] Handle errors appropriately (no silent failures)

### Before Committing
- [ ] Run tests locally: `pytest tests/`
- [ ] Run linter: `ruff check src/ tests/`
- [ ] Check types: `mypy src/`
- [ ] Update documentation if needed
- [ ] Write meaningful commit messages
- [ ] Ensure no sensitive data in commits

## 🔧 Configuration Management

### Centralized Configuration
```yaml
# All configs in configs/ directory
configs/
├── production.yaml    # Production settings
├── staging.yaml       # Staging environment
└── development.yaml   # Local development
```

### Environment Variables (.env)
```bash
# Critical settings that change per environment
MODEL_PATH=gpt-oss-20b/original
LOG_LEVEL=INFO
API_KEY=xxx  # Never commit actual keys
```

### Configuration Priority
1. Environment variables (highest priority)
2. .env file
3. Config files (configs/*.yaml)
4. Default values in code (lowest priority)

## 🚦 Feature Flags & Safety Mechanisms

### Feature Flag System
```python
# src/moe/optimization_safety/optimization_control_center.py

class OptimizationControlCenter:
    """Centralized control for all optimizations"""

    SAFE_OPTIMIZATIONS = {
        'fp16': True,      # ✅ Proven safe
        'sdpa': True,      # ✅ Tested thoroughly
        'top_k': True,     # ✅ Production ready
    }

    EXPERIMENTAL = {
        'torch_compile': False,  # ❌ Causes regression
        'int8': False,          # ⚠️ Has issues
        'dynamic_batch': False,  # 🧪 Experimental
    }
```

### Enable/Disable Pattern
```python
# Always provide safe enable/disable mechanisms
if control.is_enabled('feature_name'):
    # Feature code
else:
    # Fallback to safe default
```

### Automatic Rollback
```python
# Health monitoring with automatic rollback
monitor = HealthMonitor(thresholds={
    'latency_ms': 500,
    'memory_gb': 22,
    'error_rate': 0.01
})

if monitor.detect_degradation():
    control.rollback_to_safe_state()
```

## 🧪 Testing Requirements

### Test Coverage Targets
- Unit tests: >80% coverage
- Integration tests: All critical paths
- Performance tests: Before/after any optimization

### Test Types Required
```python
# 1. Unit Tests (test_unit.py)
def test_component():
    """Test individual components in isolation"""

# 2. Integration Tests (test_functional.py)
def test_end_to_end():
    """Test complete workflows"""

# 3. Performance Tests (test_performance.py)
def test_performance():
    """Verify performance meets requirements"""

# 4. Safety Tests
def test_rollback():
    """Ensure rollback mechanisms work"""
```

## 📊 Performance Standards

### Minimum Requirements
- Throughput: >6 tokens/second
- First token latency: <500ms
- Memory usage: <22GB (for 24GB GPU)
- Error rate: <1%

### Performance Testing
```bash
# Always benchmark before claiming improvements
python tests/test_performance.py --benchmark

# Document results with evidence
"Achieved 29.1 TPS with FP16+SDPA (tested 2024-09-23)"
```

## 🚨 Monitoring & Alerting

### Required Metrics
```python
REQUIRED_METRICS = {
    'throughput_tps': 'Tokens per second',
    'latency_ms': 'First token latency',
    'memory_gb': 'GPU memory usage',
    'error_rate': 'Request failure rate',
    'gpu_temp': 'GPU temperature'
}
```

### Health Checks
```python
@app.get('/health')
async def health_check():
    return {
        'status': 'healthy',
        'model_loaded': True,
        'gpu_available': True,
        'memory_available': True
    }
```

### Logging Standards
```python
# Use structured logging
import logging

logger = logging.getLogger(__name__)

# Good
logger.info("Model loaded", extra={
    'model_path': path,
    'load_time_s': elapsed,
    'memory_gb': memory_used
})

# Bad
print(f"Model loaded from {path}")  # Never use print
```

## 🔄 Version Control

### Branch Strategy
```bash
main/master     # Production-ready code only
├── develop     # Integration branch
├── feature/*   # New features
├── bugfix/*    # Bug fixes
└── hotfix/*    # Emergency fixes
```

### Commit Standards
```bash
# Good commit messages
feat: Add INT8 quantization support
fix: Resolve memory leak in expert cache
docs: Update performance benchmarks
test: Add rollback mechanism tests
refactor: Simplify routing logic

# Bad commit messages
"fixed stuff"
"update"
"asdf"
```

## 🐛 Debugging & Troubleshooting

### Debug Mode
```python
# Use environment variables for debug mode
if os.getenv('DEBUG') == 'true':
    logging.setLevel(logging.DEBUG)
    torch.autograd.set_detect_anomaly(True)
```

### Common Issues & Solutions
```yaml
Issue: "CUDA out of memory"
Solution:
  - Reduce batch_size
  - Clear cache: torch.cuda.empty_cache()
  - Use gradient checkpointing

Issue: "Slow inference"
Solution:
  - Verify TORCH_COMPILE_DISABLE=1
  - Check if using FP16
  - Ensure SDPA is enabled

Issue: "Model not loading"
Solution:
  - Verify model path exists
  - Check file permissions
  - Ensure sufficient memory
```

## 📝 Documentation Standards

### Required Documentation
1. **README.md** - Project overview and quick start
2. **docs/TECHNICAL.md** - Technical architecture
3. **docs/OPERATIONS.md** - Deployment and operations
4. **docs/PERFORMANCE.md** - Benchmarks and optimization
5. **docs/DEVELOPMENT.md** - Development roadmap

### Documentation Rules
- Document actual state, not aspirational
- Include timestamps for benchmarks
- Provide examples for complex features
- Keep documentation next to code
- Update docs with code changes

## 🔐 Security Practices

### Secret Management
```bash
# Never commit secrets
.env              # Local only, gitignored
.env.example      # Template with dummy values

# Use secret managers in production
AWS Secrets Manager
Azure Key Vault
HashiCorp Vault
```

### Input Validation
```python
# Always validate inputs
from pydantic import BaseModel, Field

class Request(BaseModel):
    text: str = Field(..., max_length=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
```

## ✅ Code Review Checklist

### Before Requesting Review
- [ ] Code follows project style guide
- [ ] All tests pass locally
- [ ] No hardcoded values
- [ ] Errors are handled properly
- [ ] Documentation is updated
- [ ] No sensitive data exposed
- [ ] Performance impact assessed

### Review Focus Areas
1. **Security**: No vulnerabilities introduced
2. **Performance**: No regressions
3. **Maintainability**: Code is clear and simple
4. **Testing**: Adequate test coverage
5. **Documentation**: Changes are documented

## 🚀 Deployment Checklist

### Pre-Deployment
- [ ] All tests passing in CI
- [ ] Performance benchmarks acceptable
- [ ] Configuration reviewed
- [ ] Rollback plan ready
- [ ] Monitoring configured
- [ ] Documentation updated

### Deployment Steps
```bash
# 1. Test in development
ENV=development python main.py

# 2. Test in staging
ENV=staging python main.py

# 3. Deploy to production with monitoring
ENV=production python main.py

# 4. Verify health checks
curl http://localhost:8000/health

# 5. Monitor metrics for 30 minutes
```

### Post-Deployment
- [ ] Health checks passing
- [ ] Metrics within thresholds
- [ ] No error spike in logs
- [ ] Performance matches expectations
- [ ] Document deployment results

## 🔴 Emergency Procedures

### Rollback Process
```bash
# 1. Immediate rollback if issues detected
git revert HEAD
git push

# 2. Or use feature flags
control.disable_optimization('problematic_feature')

# 3. Clear caches if needed
torch.cuda.empty_cache()
redis-cli FLUSHALL  # If using Redis
```

### Incident Response
1. **Detect**: Monitoring alerts trigger
2. **Assess**: Check logs and metrics
3. **Respond**: Rollback or hotfix
4. **Document**: Create incident report
5. **Review**: Post-mortem meeting

## 📚 Learning from Mistakes

### Past Issues (Never Repeat)
1. **torch.compile regression** - Always test optimizations thoroughly
2. **INT8 dtype mismatch** - Ensure type compatibility across layers
3. **False performance claims** - Only claim what's measured
4. **Mixed venvs** - Maintain clean environments

### Continuous Improvement
- Regular code reviews
- Weekly performance audits
- Monthly dependency updates
- Quarterly security reviews

## 🎯 Success Metrics

### Project Health Indicators
- ✅ All tests passing
- ✅ >80% code coverage
- ✅ <1% error rate in production
- ✅ Performance meets SLA
- ✅ Documentation up to date
- ✅ No critical vulnerabilities
- ✅ Feature flags working
- ✅ Rollback tested monthly

## 💡 Best Practices Summary

1. **Test everything** - No untested code in production
2. **Measure first** - Benchmark before claiming improvements
3. **Document reality** - Be honest about what works
4. **Use safety nets** - Feature flags, monitoring, rollbacks
5. **Centralize config** - One source of truth
6. **Handle errors** - No silent failures
7. **Review code** - Four eyes principle
8. **Monitor always** - Observability is crucial
9. **Keep it simple** - Complexity kills projects
10. **Learn and improve** - Post-mortems for all incidents

---

## Project-Specific Guidelines

### Integrated System Architecture
This project consists of TWO integrated components:
1. **Multi-Agent Discussion System** (main application)
2. **MoE Backend** (GPT-OSS-20B support)

### When Working on Agent System
- Main entry: `main.py`
- Core files: `src/agents/`, `src/core/`, `src/ui/`
- Keep integration with MoE backend intact
- Test with multiple model providers

### When Working on MoE Backend
- Core files: `src/moe/`
- Maintain performance: 29.1 TPS baseline
- Keep memory under 7.3GB
- Test with agent system integration

### Integration Points
```python
# src/config.py - Model provider selection
ModelProvider.GPT_OSS  # Uses MoE backend

# src/core/model_manager.py - Model loading
if provider == ModelProvider.GPT_OSS:
    # Load via MoE implementation
```

## Quick Reference Commands

```bash
# Development - Agent System
python main.py --help          # Show options
streamlit run src/ui/streamlit_app.py  # UI
python -m src.api.server       # API server

# Development - MoE Backend
pytest tests/test_performance.py  # MoE benchmarks
python tests/test_functional.py   # MoE integration

# Testing Both Systems
pytest tests/                  # Run all tests
ruff check src/                # Lint code
mypy src/                      # Type check

# Docker
docker-compose up -d           # Start services
docker-compose logs -f         # View logs
docker-compose down            # Stop services

# Git
git checkout -b feature/name   # New feature branch
git commit -m "feat: message"  # Commit with convention
git push origin feature/name   # Push branch

# Performance Monitoring
python tests/test_performance.py --benchmark
nvidia-smi                     # GPU monitoring
htop                          # System monitoring

# Debugging
export DEBUG=true              # Enable debug mode
export LOG_LEVEL=DEBUG         # Verbose logging
python -m pdb main.py          # Python debugger
```

---

**Remember**: Good engineering is about discipline, not shortcuts. Follow these guidelines and we'll maintain a professional, reliable codebase.