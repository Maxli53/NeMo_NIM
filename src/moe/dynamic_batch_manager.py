#!/usr/bin/env python3
"""
Dynamic Batching with Gradient Accumulation
Phase 1.1 Optimization from Roadmap

Automatically finds optimal batch size for available memory
and implements gradient accumulation for effective larger batches.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
import logging
from dataclasses import dataclass
import psutil
import GPUtil
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for dynamic batching"""
    min_batch_size: int = 1
    max_batch_size: int = 256
    memory_margin: float = 0.15  # Keep 15% memory free
    gradient_accumulation_steps: int = 1
    enable_auto_tuning: bool = True
    profile_steps: int = 10
    

class DynamicBatchManager:
    """
    Manages dynamic batch sizing and gradient accumulation
    for optimal memory utilization and throughput.
    """
    
    def __init__(self, model: nn.Module, config: BatchConfig = None):
        self.model = model
        self.config = config or BatchConfig()
        self.device = next(model.parameters()).device
        
        # Tracking metrics
        self.optimal_batch_size = None
        self.memory_stats = []
        self.throughput_stats = []
        self.current_accumulation_steps = 1
        
        # Memory monitoring
        self.baseline_memory = self._get_current_memory()
        logger.info(f"Initialized DynamicBatchManager with baseline memory: {self.baseline_memory:.2f}GB")
        
    def _get_current_memory(self) -> float:
        """Get current GPU memory usage in GB"""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) / 1e9
        return psutil.Process().memory_info().rss / 1e9
        
    def _get_available_memory(self) -> float:
        """Get available GPU memory in GB"""
        if self.device.type == 'cuda':
            total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            
            # Account for fragmentation
            available = total - reserved
            usable = available * (1 - self.config.memory_margin)
            return max(0, usable)
        
        # CPU fallback
        return psutil.virtual_memory().available / 1e9 * (1 - self.config.memory_margin)
        
    def can_fit_batch(self, batch_size: int, seq_length: int = 512) -> bool:
        """
        Test if a batch size can fit in memory without OOM.
        Uses a dummy forward pass to measure memory requirements.
        """
        torch.cuda.empty_cache()
        available = self._get_available_memory()
        
        try:
            # Create dummy batch
            dummy_input = torch.randn(
                batch_size, seq_length, 
                self.model.config.hidden_size if hasattr(self.model, 'config') else 2880,
                device=self.device,
                dtype=torch.float16
            )
            
            # Test forward pass
            with torch.no_grad():
                _ = self.model(dummy_input)
                
            # Check memory usage
            used = self._get_current_memory() - self.baseline_memory
            
            # Clean up
            del dummy_input
            torch.cuda.empty_cache()
            
            # Check if we're within limits
            return used < available
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.debug(f"Batch size {batch_size} caused OOM: {e}")
            torch.cuda.empty_cache()
            return False
            
    def auto_tune_batch_size(self) -> int:
        """
        Automatically find optimal batch size using binary search.
        Returns the maximum batch size that fits in memory.
        """
        logger.info("Starting batch size auto-tuning...")
        
        low = self.config.min_batch_size
        high = self.config.max_batch_size
        optimal = self.config.min_batch_size
        
        # Binary search for optimal batch size
        while low <= high:
            mid = (low + high) // 2
            
            if self.can_fit_batch(mid):
                optimal = mid
                low = mid + 1
                logger.debug(f"Batch size {mid} fits, trying larger...")
            else:
                high = mid - 1
                logger.debug(f"Batch size {mid} too large, trying smaller...")
                
        self.optimal_batch_size = optimal
        logger.info(f"Optimal batch size found: {optimal}")
        
        # Calculate gradient accumulation steps for effective batch size
        if optimal < self.config.max_batch_size:
            self.current_accumulation_steps = min(
                self.config.max_batch_size // optimal,
                self.config.gradient_accumulation_steps
            )
            logger.info(f"Using {self.current_accumulation_steps} gradient accumulation steps")
            
        return optimal
        
    @contextmanager
    def adaptive_batch_context(self, target_batch_size: int):
        """
        Context manager for adaptive batching with automatic OOM recovery.
        """
        actual_batch_size = target_batch_size
        accumulation_steps = 1
        
        # Check if target batch size fits
        if not self.can_fit_batch(target_batch_size):
            # Find smaller batch size that fits
            actual_batch_size = self.auto_tune_batch_size()
            
            # Calculate accumulation steps to match effective batch size
            accumulation_steps = max(1, target_batch_size // actual_batch_size)
            
        try:
            yield actual_batch_size, accumulation_steps
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.warning(f"OOM encountered with batch size {actual_batch_size}: {e}")
            torch.cuda.empty_cache()
            
            # Retry with smaller batch
            smaller_batch = max(1, actual_batch_size // 2)
            logger.info(f"Retrying with batch size {smaller_batch}")
            
            yield smaller_batch, accumulation_steps * 2
            
    def optimize_gradient_accumulation(self, loss: torch.Tensor, step: int) -> torch.Tensor:
        """
        Optimize gradient accumulation for effective larger batch sizes.
        """
        # Scale loss by accumulation steps
        if self.current_accumulation_steps > 1:
            loss = loss / self.current_accumulation_steps
            
        # Accumulate gradients
        loss.backward()
        
        # Only step optimizer after accumulation
        if (step + 1) % self.current_accumulation_steps == 0:
            return loss * self.current_accumulation_steps  # Return unscaled loss for logging
            
        return loss
        
    def profile_batch_performance(self, batch_sizes: List[int] = None) -> Dict[int, Dict]:
        """
        Profile performance across different batch sizes.
        Returns throughput and memory usage statistics.
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            
        results = {}
        
        for batch_size in batch_sizes:
            if not self.can_fit_batch(batch_size):
                logger.info(f"Batch size {batch_size} exceeds memory, skipping...")
                continue
                
            # Measure throughput
            torch.cuda.synchronize()
            start_time = time.time()
            
            dummy_input = torch.randn(
                batch_size, 512, 
                self.model.config.hidden_size if hasattr(self.model, 'config') else 2880,
                device=self.device,
                dtype=torch.float16
            )
            
            with torch.no_grad():
                for _ in range(self.config.profile_steps):
                    _ = self.model(dummy_input)
                    
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            # Calculate metrics
            throughput = (batch_size * self.config.profile_steps) / elapsed
            memory_used = self._get_current_memory() - self.baseline_memory
            
            results[batch_size] = {
                'throughput_samples_per_sec': throughput,
                'memory_gb': memory_used,
                'time_per_batch_ms': (elapsed / self.config.profile_steps) * 1000
            }
            
            # Clean up
            del dummy_input
            torch.cuda.empty_cache()
            
            logger.info(f"Batch {batch_size}: {throughput:.1f} samples/sec, {memory_used:.2f}GB")
            
        return results
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        if self.device.type == 'cuda':
            return {
                'allocated_gb': torch.cuda.memory_allocated(self.device) / 1e9,
                'reserved_gb': torch.cuda.memory_reserved(self.device) / 1e9,
                'free_gb': self._get_available_memory(),
                'optimal_batch_size': self.optimal_batch_size or 'Not tuned',
                'accumulation_steps': self.current_accumulation_steps
            }
        return {
            'ram_gb': psutil.Process().memory_info().rss / 1e9,
            'available_gb': psutil.virtual_memory().available / 1e9
        }
        

class GradientAccumulator:
    """
    Helper class for gradient accumulation with mixed precision support.
    """
    
    def __init__(self, model: nn.Module, accumulation_steps: int = 1):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
        self.accumulated_loss = 0
        
    def accumulate(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> bool:
        """
        Accumulate gradients and return True when optimizer should step.
        """
        # Scale loss
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.accumulated_loss += loss.item()
        self.step_count += 1
        
        # Check if should step
        if self.step_count >= self.accumulation_steps:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Step optimizer
            optimizer.step()
            optimizer.zero_grad()
            
            # Reset
            avg_loss = self.accumulated_loss / self.accumulation_steps
            self.step_count = 0
            self.accumulated_loss = 0
            
            return True, avg_loss
            
        return False, None


if __name__ == "__main__":
    # Test dynamic batching
    logging.basicConfig(level=logging.INFO)
    
    # Mock model for testing
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('Config', (), {'hidden_size': 2880})
            self.linear = nn.Linear(2880, 2880)
            
        def forward(self, x):
            return self.linear(x)
    
    model = MockModel().cuda()
    manager = DynamicBatchManager(model)
    
    # Auto-tune batch size
    optimal = manager.auto_tune_batch_size()
    print(f"Optimal batch size: {optimal}")
    
    # Profile different batch sizes
    results = manager.profile_batch_performance([1, 2, 4, 8, 16, 32])
    
    # Print results
    print("\nBatch Size Performance:")
    for batch_size, metrics in results.items():
        print(f"  {batch_size}: {metrics['throughput_samples_per_sec']:.1f} samples/sec")
    
    # Memory stats
    stats = manager.get_memory_stats()
    print(f"\nMemory Stats: {stats}")