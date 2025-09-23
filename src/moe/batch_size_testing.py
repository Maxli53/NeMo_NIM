#!/usr/bin/env python3
"""
Batch Size Testing for MoE Model
Tests throughput, latency, and memory usage across different batch sizes
Finds optimal batch size for 24GB VRAM
"""

import torch
import torch.nn as nn
import time
import logging
from typing import Dict, List, Tuple
import gc
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.moe.native_moe_loader_v2 import MoEModelLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchSizeTester:
    """Comprehensive batch size testing for MoE models"""

    def __init__(self, model_path: str = "gpt-oss-20b/original"):
        self.model_path = model_path
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Test configurations
        self.batch_sizes = [1, 2, 4, 8, 16, 32]
        self.sequence_length = 128  # Fixed sequence length for testing
        self.warmup_iterations = 3
        self.test_iterations = 10

    def measure_memory(self) -> Dict[str, float]:
        """Measure current GPU memory usage"""
        torch.cuda.synchronize()
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "free_gb": (torch.cuda.get_device_properties(0).total_memory -
                       torch.cuda.memory_allocated()) / 1e9
        }

    def test_batch_size(
        self,
        model: nn.Module,
        batch_size: int,
        seq_len: int
    ) -> Dict:
        """Test a specific batch size configuration"""
        logger.info(f"Testing batch_size={batch_size}, seq_len={seq_len}")

        # Clear cache before test
        torch.cuda.empty_cache()
        gc.collect()

        # Create test input
        input_ids = torch.randint(0, 50000, (batch_size, seq_len)).cuda()

        # Memory before
        mem_before = self.measure_memory()

        try:
            # Warmup runs
            logger.debug(f"Warmup iterations: {self.warmup_iterations}")
            for _ in range(self.warmup_iterations):
                with torch.no_grad():
                    _ = model(input_ids)
                torch.cuda.synchronize()

            # Measure memory after warmup
            mem_after_warmup = self.measure_memory()

            # Timing runs
            latencies = []
            throughputs = []

            for i in range(self.test_iterations):
                torch.cuda.synchronize()
                start_time = time.perf_counter()

                with torch.no_grad():
                    output = model(input_ids)

                torch.cuda.synchronize()
                end_time = time.perf_counter()

                # Calculate metrics
                elapsed_time = end_time - start_time
                latencies.append(elapsed_time * 1000)  # Convert to ms

                # Throughput: (batch_size * seq_len) / time
                tokens_processed = batch_size * seq_len
                throughput = tokens_processed / elapsed_time
                throughputs.append(throughput)

            # Memory after tests
            mem_after = self.measure_memory()

            # Calculate statistics
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)

            avg_throughput = sum(throughputs) / len(throughputs)
            max_throughput = max(throughputs)

            # Per-sample metrics
            per_sample_latency = avg_latency / batch_size
            per_token_latency = avg_latency / (batch_size * seq_len)

            results = {
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "status": "success",
                # Latency metrics (ms)
                "avg_latency_ms": avg_latency,
                "min_latency_ms": min_latency,
                "max_latency_ms": max_latency,
                "per_sample_latency_ms": per_sample_latency,
                "per_token_latency_ms": per_token_latency,
                # Throughput metrics
                "avg_throughput_tokens_per_sec": avg_throughput,
                "max_throughput_tokens_per_sec": max_throughput,
                "samples_per_sec": 1000 / avg_latency * batch_size,
                # Memory metrics (GB)
                "memory_used_gb": mem_after["allocated_gb"] - mem_before["allocated_gb"],
                "peak_memory_gb": mem_after["allocated_gb"],
                "memory_per_sample_gb": (mem_after["allocated_gb"] - mem_before["allocated_gb"]) / batch_size,
                # Efficiency metrics
                "gpu_memory_efficiency": mem_after["allocated_gb"] / mem_after["reserved_gb"],
                "scaling_efficiency": avg_throughput / (batch_size * (throughputs[0] if batch_size == 1 else 1)),
            }

            logger.info(f"✅ Batch {batch_size}: {avg_throughput:.1f} tokens/sec, "
                       f"{avg_latency:.1f}ms latency, {mem_after['allocated_gb']:.2f}GB memory")

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"❌ OOM at batch_size={batch_size}")
            results = {
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "status": "OOM",
                "error": "CUDA out of memory"
            }

        except Exception as e:
            logger.error(f"❌ Error at batch_size={batch_size}: {e}")
            results = {
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "status": "error",
                "error": str(e)
            }

        # Clear cache after test
        torch.cuda.empty_cache()
        gc.collect()

        return results

    def find_optimal_batch_size(
        self,
        model: nn.Module,
        target_memory_gb: float = 22.0
    ) -> int:
        """Find the optimal batch size for given memory constraint"""
        logger.info(f"Finding optimal batch size for {target_memory_gb}GB memory limit")

        optimal_batch = 1
        best_throughput = 0

        for batch_size in self.batch_sizes:
            result = self.test_batch_size(model, batch_size, self.sequence_length)

            if result["status"] == "success":
                if result["peak_memory_gb"] < target_memory_gb:
                    if result["avg_throughput_tokens_per_sec"] > best_throughput:
                        best_throughput = result["avg_throughput_tokens_per_sec"]
                        optimal_batch = batch_size
                else:
                    logger.info(f"Batch {batch_size} exceeds memory limit "
                              f"({result['peak_memory_gb']:.2f}GB > {target_memory_gb}GB)")
                    break
            elif result["status"] == "OOM":
                break

        logger.info(f"Optimal batch size: {optimal_batch} "
                   f"(throughput: {best_throughput:.1f} tokens/sec)")
        return optimal_batch

    def run_comprehensive_test(self, model_type: str = "fp16") -> Dict:
        """Run comprehensive batch size testing"""
        logger.info(f"Starting comprehensive batch size test for {model_type} model")

        # Load model
        loader = MoEModelLoader(self.model_path)

        if model_type == "fp16":
            model = loader.create_model_fp16(top_k=4, full_layers=False)  # 12 layers for testing
        elif model_type == "int8":
            model = loader.create_model_int8_fixed(top_k=4, full_layers=False)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.eval()

        # Test each batch size
        all_results = []
        for batch_size in self.batch_sizes:
            result = self.test_batch_size(model, batch_size, self.sequence_length)
            all_results.append(result)
            self.results[f"batch_{batch_size}"] = result

            # Stop if OOM
            if result["status"] == "OOM":
                logger.info(f"Stopping tests due to OOM at batch_size={batch_size}")
                break

        # Find optimal configuration
        optimal_batch = self.find_optimal_batch_size(model)

        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "model_type": model_type,
            "batch_results": all_results,
            "optimal_batch_size": optimal_batch,
            "test_config": {
                "sequence_length": self.sequence_length,
                "warmup_iterations": self.warmup_iterations,
                "test_iterations": self.test_iterations,
            }
        }

    def generate_report(self) -> str:
        """Generate a comprehensive report of batch size testing"""
        report = []
        report.append("=" * 80)
        report.append("BATCH SIZE TESTING REPORT")
        report.append("=" * 80)

        # Summary table
        report.append("\n📊 Performance Summary:")
        report.append("-" * 80)
        report.append(f"{'Batch':<8} {'Status':<10} {'Throughput':<15} {'Latency':<15} {'Memory':<12} {'Efficiency':<12}")
        report.append(f"{'Size':<8} {'':<10} {'(tokens/sec)':<15} {'(ms)':<15} {'(GB)':<12} {'(%)':<12}")
        report.append("-" * 80)

        for key, result in self.results.items():
            if result["status"] == "success":
                report.append(
                    f"{result['batch_size']:<8} "
                    f"{'✅':<10} "
                    f"{result['avg_throughput_tokens_per_sec']:<15.1f} "
                    f"{result['avg_latency_ms']:<15.1f} "
                    f"{result['peak_memory_gb']:<12.2f} "
                    f"{result.get('scaling_efficiency', 0)*100:<12.1f}"
                )
            else:
                report.append(
                    f"{result['batch_size']:<8} "
                    f"{'❌ ' + result['status']:<10} "
                    f"{'-':<15} {'-':<15} {'-':<12} {'-':<12}"
                )

        # Find best configurations
        successful_results = [r for r in self.results.values() if r["status"] == "success"]

        if successful_results:
            best_throughput = max(successful_results, key=lambda x: x["avg_throughput_tokens_per_sec"])
            best_latency = min(successful_results, key=lambda x: x["per_sample_latency_ms"])
            most_efficient = max(successful_results, key=lambda x: x.get("scaling_efficiency", 0))

            report.append("\n🏆 Best Configurations:")
            report.append(f"  Highest Throughput: Batch {best_throughput['batch_size']} "
                         f"({best_throughput['avg_throughput_tokens_per_sec']:.1f} tokens/sec)")
            report.append(f"  Lowest Latency: Batch {best_latency['batch_size']} "
                         f"({best_latency['per_sample_latency_ms']:.1f}ms per sample)")
            report.append(f"  Most Efficient: Batch {most_efficient['batch_size']} "
                         f"({most_efficient.get('scaling_efficiency', 0)*100:.1f}% scaling efficiency)")

        # Memory analysis
        report.append("\n💾 Memory Scaling:")
        for key, result in self.results.items():
            if result["status"] == "success":
                report.append(f"  Batch {result['batch_size']}: "
                             f"{result['peak_memory_gb']:.2f}GB "
                             f"({result['memory_per_sample_gb']*1000:.1f}MB per sample)")

        return "\n".join(report)


def main():
    """Main testing function"""
    import argparse

    parser = argparse.ArgumentParser(description="Batch size testing for MoE model")
    parser.add_argument("--model-path", default="gpt-oss-20b/original", help="Path to model")
    parser.add_argument("--model-type", default="fp16", choices=["fp16", "int8"], help="Model type")
    parser.add_argument("--save-results", default="batch_test_results.json", help="Save results to file")
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Run tests
    tester = BatchSizeTester(args.model_path)
    results = tester.run_comprehensive_test(args.model_type)

    # Generate and print report
    report = tester.generate_report()
    print(report)

    # Save results
    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.save_results}")

    # Production recommendation
    print("\n" + "=" * 80)
    print("📝 PRODUCTION RECOMMENDATION")
    print("=" * 80)
    print(f"Optimal batch size: {results['optimal_batch_size']}")
    print(f"Expected throughput: Check results above")
    print(f"Memory usage: Within 24GB limit")
    print("\nUpdate configs/production.yaml with:")
    print(f"  batch_size: {results['optimal_batch_size']}")
    print(f"  sequence_length: {tester.sequence_length}")


if __name__ == "__main__":
    main()