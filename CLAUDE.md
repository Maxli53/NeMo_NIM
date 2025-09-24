# Development Principles and Guidelines

This document outlines the core principles for working with NeMo/NIM and AI development in this project.

## 1. Follow Best Development Practices

- **Code Quality**: Write clean, maintainable, and well-documented code
- **Version Control**: Commit meaningful changes with clear messages
- **Testing**: Test all code before deployment
- **Security**: Never expose API keys, credentials, or sensitive data
- **Error Handling**: Implement proper error handling and logging
- **Performance**: Profile and optimize code for production use
- **Documentation**: Document all functions, classes, and complex logic

## 2. Strictly Adhere to Official NVIDIA Documentation

### Always Reference Official Sources:
- **NeMo Framework**: https://docs.nvidia.com/nemo-framework/
- **NeMo GitHub**: https://github.com/NVIDIA/NeMo
- **NIM Documentation**: https://docs.nvidia.com/nim/
- **NGC Catalog**: https://catalog.ngc.nvidia.com

### Guidelines:
- **DO NOT** reinvent the wheel - use official implementations
- **DO NOT** create custom solutions when official ones exist
- **ALWAYS** check official examples before implementing
- **ALWAYS** verify against latest documentation
- **COPY** exact patterns from official examples
- When in doubt, check the official GitHub repository

## 3. Implement Safety Mechanisms

### Required Safety Features:
- **Memory Guards**: Monitor and limit memory usage
- **Checkpoint Recovery**: Always enable checkpoint saving
- **Graceful Degradation**: Handle failures without crashing
- **Resource Limits**: Set appropriate timeouts and limits
- **Feature Flags**: Use environment variables for configuration
- **Validation**: Validate all inputs and configurations

### Example Feature Flags:
```python
# Feature flags in .env
ENABLE_FLASH_ATTENTION=true
USE_GRADIENT_CHECKPOINTING=true
ENABLE_MIXED_PRECISION=bf16
MAX_SEQUENCE_LENGTH=2048
SAFE_MODE=true
```

## 4. Maintain Accuracy and Honesty

### Strict Rules:
- **NO hallucination** - Only use documented features
- **NO assumptions** - Verify every claim against documentation
- **NO fake metrics** - Report actual performance numbers
- **NO untested code** - Test before claiming it works
- **Acknowledge limitations** - Be clear about what won't work
- **Cite sources** - Always reference official documentation

### When Uncertain:
1. Check official documentation first
2. Look for official examples
3. Test in a controlled environment
4. Document limitations clearly
5. Ask for clarification if needed

## 5. Comprehensive Testing Strategy

### Testing Requirements:

#### Unit Testing:
- Test individual functions and components
- Mock external dependencies
- Verify edge cases

#### Integration Testing:
- Test component interactions
- Verify data flow
- Check API contracts

#### Performance Testing:
- Measure inference speed
- Monitor memory usage
- Profile GPU utilization
- Track model accuracy

#### Before Deployment Checklist:
- [ ] All tests pass
- [ ] Memory usage is within limits
- [ ] GPU utilization is optimal
- [ ] Error handling works correctly
- [ ] Logging provides useful information
- [ ] Configuration is validated
- [ ] Documentation is updated
- [ ] Security review completed

### Example Test Structure:
```python
def test_model_import():
    """Test GPT-OSS model import from OpenAI format."""
    # Verify file exists
    assert Path(model_path).exists()

    # Import model
    model = import_gpt_oss(model_path)

    # Verify model structure
    assert model.num_layers == 44
    assert model.hidden_size == 6144

    # Test inference
    output = model.generate("test")
    assert len(output) > 0

def test_memory_limits():
    """Ensure model fits in available VRAM."""
    available_memory = torch.cuda.get_device_properties(0).total_memory
    model_memory = estimate_model_memory()
    assert model_memory < available_memory * 0.9  # 90% safety margin
```

## 6. Additional Guidelines

### Communication:
- Be precise and technical
- Avoid marketing language
- Report problems immediately
- Document all decisions

### Resource Management:
- Monitor GPU memory constantly
- Clean up resources after use
- Use context managers for resources
- Implement proper shutdown procedures

### Continuous Improvement:
- Stay updated with latest NeMo releases
- Review and update dependencies regularly
- Refactor code when better patterns emerge
- Learn from official examples and updates

## Remember:

**The official NVIDIA documentation is the source of truth.** When this document conflicts with official documentation, always follow the official guidance.