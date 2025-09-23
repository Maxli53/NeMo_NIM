#!/usr/bin/env python3
"""
Fast model loader with progress indication and optimizations
"""

import torch
import time
import logging
from pathlib import Path
from typing import Optional
import sys

from .gpt_oss_model import GPTOSSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgressBar:
    """Simple progress bar for terminal output"""

    def __init__(self, total: int, prefix: str = "Progress"):
        self.total = total
        self.prefix = prefix
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        self.current += n
        percent = self.current * 100 // self.total
        elapsed = time.time() - self.start_time

        # Simple progress indicator
        bar_length = 40
        filled = bar_length * self.current // self.total
        bar = '=' * filled + '-' * (bar_length - filled)

        sys.stdout.write(f'\r{self.prefix}: [{bar}] {percent}% ({self.current}/{self.total}) - {elapsed:.1f}s')
        sys.stdout.flush()

        if self.current >= self.total:
            print()  # New line when complete


class FastGPTOSSModel(GPTOSSModel):
    """Optimized model loader with progress tracking"""

    def __init__(
        self,
        model_path: str = "gpt-oss-20b/original",
        load_weights: bool = True,
        num_layers: Optional[int] = None,  # Load only first N layers for testing
        show_progress: bool = True
    ):
        self.show_progress = show_progress
        self.num_layers_to_load = num_layers

        # Time the initialization
        start_time = time.time()

        if self.show_progress:
            print("Initializing GPT-OSS-20B Model...")
            print("-" * 60)

        # Call parent init (but override weight loading)
        self._load_weights_on_init = load_weights
        super().__init__(model_path)

        init_time = time.time() - start_time

        if self.show_progress:
            print("-" * 60)
            print(f"[OK] Model initialized in {init_time:.1f} seconds")
            print(f"  - Parameters: {sum(p.numel() for p in self.parameters())/1e9:.2f}B")
            print(f"  - Memory used: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    def _load_weights(self):
        """Optimized weight loading with progress tracking"""

        if not hasattr(self, '_load_weights_on_init') or not self._load_weights_on_init:
            logger.info("Skipping weight loading (lazy mode)")
            return

        from safetensors import safe_open
        weights_path = self.model_path / "model.safetensors"

        if self.show_progress:
            print(f"\nLoading weights from: {weights_path.name}")

        # Count steps for progress bar
        num_layers = self.num_layers_to_load or len(self.blocks)
        total_steps = num_layers + 3  # blocks + embedding + norm + lm_head

        if self.show_progress:
            progress = ProgressBar(total_steps, "Loading")

        with safe_open(weights_path, framework="pt", device="cpu") as f:
            # Load embeddings
            if "embedding.weight" in f.keys():
                self.embedding.weight.data = f.get_tensor("embedding.weight").cuda()
                if self.show_progress:
                    progress.update()
                else:
                    logger.info(f"Loaded embeddings: {self.embedding.weight.shape}")

            # Load transformer blocks
            for i in range(num_layers):
                self.blocks[i].load_from_safetensors(f, i)
                if self.show_progress:
                    progress.update()
                elif i % 6 == 0:
                    logger.info(f"Loaded block {i+1}/{num_layers}")

            # Skip remaining blocks if partial load
            if num_layers < len(self.blocks):
                logger.warning(f"Loaded only {num_layers}/{len(self.blocks)} blocks")

            # Load final layers
            if "norm.scale" in f.keys():
                self.ln_f.load_from_state_dict(f.get_tensor("norm.scale").cuda())
                if self.show_progress:
                    progress.update()

            if "unembedding.weight" in f.keys():
                self.lm_head.weight.data = f.get_tensor("unembedding.weight").cuda()
                if self.show_progress:
                    progress.update()
            elif "lm_head.weight" in f.keys():
                self.lm_head.weight.data = f.get_tensor("lm_head.weight").cuda()
            else:
                self.lm_head.weight = self.embedding.weight

        if self.show_progress:
            print("\n[OK] All weights loaded successfully")

    @classmethod
    def load_fast(cls, model_path: str = "gpt-oss-20b/original", **kwargs):
        """Convenience method for fast loading"""
        return cls(model_path, show_progress=True, **kwargs)

    @classmethod
    def load_test(cls, model_path: str = "gpt-oss-20b/original"):
        """Load minimal model for testing (only first 3 layers)"""
        print("Loading test model (3 layers only)...")
        return cls(model_path, num_layers=3, show_progress=True)


def benchmark_loading():
    """Benchmark different loading strategies"""

    print("=" * 60)
    print("Benchmarking Model Loading Strategies")
    print("=" * 60)

    # Test 1: Fast load with progress
    print("\n1. Fast loader with progress bar:")
    start = time.time()
    model = FastGPTOSSModel.load_fast()
    fast_time = time.time() - start
    del model
    torch.cuda.empty_cache()

    # Test 2: Test model (3 layers)
    print("\n2. Test model (3 layers only):")
    start = time.time()
    test_model = FastGPTOSSModel.load_test()
    test_time = time.time() - start

    # Test forward pass
    print("\n3. Testing forward pass...")
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).cuda()
    start = time.time()
    with torch.no_grad():
        output = test_model(input_ids)
    forward_time = time.time() - start

    print(f"Forward pass: {forward_time:.2f}s")
    print(f"Output shape: {output['logits'].shape if isinstance(output, dict) else output.shape}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Full model load: {fast_time:.1f}s")
    print(f"  Test model load: {test_time:.1f}s")
    print(f"  Forward pass: {forward_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_loading()