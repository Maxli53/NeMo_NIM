#!/usr/bin/env python3
"""
INT8 Quantization with Proper Dtype Handling
Fixes the dtype mismatch issue: "mat1 and mat2 must have the same dtype"

Key fix: Ensure inputs are properly converted to FP32 before INT8 operations
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Tuple
import bitsandbytes as bnb

logger = logging.getLogger(__name__)


class Int8LinearFixed(nn.Module):
    """
    Fixed INT8 Linear layer that handles dtype conversion properly
    Wraps bitsandbytes Linear8bitLt with automatic dtype handling
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        has_fp16_weights: bool = False,
        threshold: float = 6.0,
        index: Optional[int] = None,
    ):
        super().__init__()

        # Create the INT8 linear layer
        self.int8_linear = bnb.nn.Linear8bitLt(
            in_features,
            out_features,
            bias=bias,
            has_fp16_weights=has_fp16_weights,
            threshold=threshold,
            index=index,
        )

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic dtype conversion

        Key fix: Convert input to FP32 if needed before INT8 operation
        """
        original_dtype = input.dtype
        device = input.device

        # CRITICAL FIX: Ensure input is FP32 for INT8 operations
        if input.dtype != torch.float32:
            input = input.float()  # Convert to FP32

        # Run through INT8 layer
        output = self.int8_linear(input)

        # Convert back to original dtype if needed (usually FP16)
        if original_dtype != torch.float32 and original_dtype in [torch.float16, torch.bfloat16]:
            output = output.to(original_dtype)

        return output


class Int8MoEWrapper:
    """
    Wrapper for converting MoE model to INT8 with proper dtype handling
    """

    @staticmethod
    def convert_to_int8(model: nn.Module, threshold: float = 6.0) -> nn.Module:
        """
        Convert model to INT8 with fixed dtype handling

        Args:
            model: Model to convert (should be in FP32)
            threshold: Outlier threshold for INT8 quantization

        Returns:
            Model with INT8 linear layers
        """
        # First ensure model is FP32
        if next(model.parameters()).dtype != torch.float32:
            logger.info("Converting model to FP32 for INT8 quantization...")
            model = model.float()

        def replace_with_int8(module: nn.Module, prefix: str = ""):
            """Recursively replace Linear layers with INT8 version"""
            for name, child in list(module.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name

                if isinstance(child, nn.Linear):
                    # Skip certain layers that should stay in FP32/FP16
                    if any(skip in full_name for skip in ['router', 'embed', 'norm', 'head']):
                        logger.debug(f"Skipping {full_name} (critical layer)")
                        continue

                    logger.debug(f"Converting {full_name} to INT8")

                    # Create INT8 replacement
                    int8_layer = Int8LinearFixed(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        threshold=threshold,
                    )

                    # Copy weights (ensure FP32)
                    with torch.no_grad():
                        int8_layer.int8_linear.weight = bnb.nn.Int8Params(
                            child.weight.data.cpu().float(),
                            requires_grad=False,
                        )

                        if child.bias is not None:
                            int8_layer.int8_linear.bias = nn.Parameter(
                                child.bias.data.float().cuda(),
                                requires_grad=False,
                            )

                    # Move to CUDA
                    int8_layer = int8_layer.cuda()

                    # Replace module
                    setattr(module, name, int8_layer)

                    # Clear memory
                    del child
                    torch.cuda.empty_cache()

                elif hasattr(child, 'children'):
                    # Recurse into containers
                    replace_with_int8(child, full_name)

        # Apply INT8 conversion
        replace_with_int8(model)

        # Convert non-linear layers back to FP16 for efficiency
        for module in model.modules():
            if not isinstance(module, (Int8LinearFixed, bnb.nn.Linear8bitLt)):
                if hasattr(module, 'half'):
                    module.half()

        logger.info("INT8 conversion complete with dtype fix")
        return model

    @staticmethod
    def verify_int8_conversion(model: nn.Module) -> Tuple[int, int, float]:
        """
        Verify INT8 conversion and calculate memory savings

        Returns:
            (num_int8_layers, total_layers, memory_saved_gb)
        """
        int8_count = 0
        total_linear = 0
        memory_saved = 0

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, Int8LinearFixed, bnb.nn.Linear8bitLt)):
                total_linear += 1

                if isinstance(module, (Int8LinearFixed, bnb.nn.Linear8bitLt)):
                    int8_count += 1
                    # Estimate memory savings (FP32 -> INT8 = 4x reduction)
                    if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                        param_count = module.in_features * module.out_features
                        memory_saved += param_count * 3 / 1e9  # 3 bytes saved per param
                elif isinstance(module, nn.Linear):
                    # This stayed FP32/FP16
                    logger.debug(f"Layer {name} not converted to INT8")

        logger.info(f"INT8 Conversion: {int8_count}/{total_linear} layers converted")
        logger.info(f"Estimated memory saved: {memory_saved:.2f} GB")

        return int8_count, total_linear, memory_saved


def test_int8_dtype_fix():
    """Test the INT8 dtype fix"""
    import time

    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
            self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
            self.norm = nn.LayerNorm(hidden_size)

        def forward(self, x):
            # This will test dtype conversion
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = self.norm(x)
            return x

    logger.info("Creating test model...")
    model = TestModel().cuda()

    # Test input (FP16 - this would normally cause dtype mismatch)
    batch_size, seq_len, hidden_size = 2, 128, 768
    input_fp16 = torch.randn(batch_size, seq_len, hidden_size).half().cuda()

    # Test original model
    logger.info("Testing original FP16 model...")
    with torch.no_grad():
        start = time.time()
        output_orig = model.half()(input_fp16)
        orig_time = time.time() - start
    logger.info(f"Original forward: {orig_time*1000:.2f}ms")

    # Convert to INT8 with fix
    logger.info("Converting to INT8 with dtype fix...")
    model_int8 = Int8MoEWrapper.convert_to_int8(model)

    # Verify conversion
    int8_count, total_count, mem_saved = Int8MoEWrapper.verify_int8_conversion(model_int8)

    # Test INT8 model (should handle FP16 input without error)
    logger.info("Testing INT8 model with FP16 input...")
    with torch.no_grad():
        start = time.time()
        output_int8 = model_int8(input_fp16)
        int8_time = time.time() - start
    logger.info(f"INT8 forward: {int8_time*1000:.2f}ms")

    # Check output similarity
    if output_orig.shape == output_int8.shape:
        diff = (output_orig - output_int8).abs().mean().item()
        logger.info(f"Output difference: {diff:.6f}")
        if diff < 0.1:
            logger.info("✅ INT8 conversion successful with minimal accuracy loss")
        else:
            logger.warning(f"⚠️ Large output difference: {diff}")

    # Memory usage
    mem_before = torch.cuda.max_memory_allocated() / 1e9
    logger.info(f"Peak GPU memory: {mem_before:.2f} GB")

    return model_int8


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_int8_dtype_fix()