#!/usr/bin/env python3
"""
Optimization Health Monitoring System
Continuous monitoring of all optimization metrics with automatic alerts
and rollback triggers.
"""

import time
import torch
import psutil
import GPUtil
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import threading
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MetricThresholds:
    """Thresholds for different metrics"""
    # Performance
    max_latency_ms: float = 200.0           # Maximum latency
    max_latency_increase: float = 1.2       # 20% increase max
    min_throughput: float = 50.0            # Min samples/sec
    
    # Quality  
    min_token_accuracy: float = 0.98        # 98% minimum
    max_perplexity_increase: float = 1.02   # 2% max increase
    max_loss_increase: float = 1.05         # 5% max increase
    
    # Stability
    max_error_rate: float = 0.01            # 1% max errors
    max_crash_rate: float = 0.001           # 0.1% max crashes
    max_timeout_rate: float = 0.05          # 5% max timeouts
    
    # Resources
    max_memory_gb: float = 8.0              # Max GPU memory
    max_memory_growth_gb_per_hour: float = 0.1  # Memory leak detection
    max_cpu_percent: float = 80.0           # CPU usage
    

@dataclass
class OptimizationMetrics:
    """Metrics for a single optimization"""
    name: str
    timestamp: float = field(default_factory=time.time)
    
    # Performance metrics
    latency_ms: float = 0.0
    throughput: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    
    # Quality metrics
    token_accuracy: float = 1.0
    perplexity: float = 0.0
    loss: float = 0.0
    
    # Stability metrics
    error_count: int = 0
    crash_count: int = 0
    timeout_count: int = 0
    total_requests: int = 0
    
    # Resource metrics
    memory_gb: float = 0.0
    cpu_percent: float = 0.0
    gpu_utilization: float = 0.0
    
    # Comparative metrics (vs baseline)
    latency_ratio: float = 1.0
    throughput_ratio: float = 1.0
    accuracy_delta: float = 0.0
    perplexity_ratio: float = 1.0
    

class OptimizationHealthMonitor:
    """
    Real-time health monitoring for all optimizations.
    Tracks metrics, detects anomalies, and triggers alerts.
    """
    
    def __init__(self, window_size: int = 100, alert_callback: Optional[Callable] = None):
        self.window_size = window_size
        self.alert_callback = alert_callback
        
        # Metrics storage
        self.current_metrics: Dict[str, OptimizationMetrics] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_metrics: Dict[str, OptimizationMetrics] = {}
        
        # Thresholds
        self.thresholds = MetricThresholds()
        
        # Alert tracking
        self.alert_history: List[Dict] = []
        self.breach_counts: Dict[str, int] = defaultdict(int)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.start_time = time.time()
        self.total_measurements = 0
        
        logger.info("Health Monitor initialized")
        
    def record_metrics(
        self,
        optimization: str,
        latency_ms: Optional[float] = None,
        throughput: Optional[float] = None,
        accuracy: Optional[float] = None,
        error: bool = False,
        crash: bool = False,
        timeout: bool = False
    ):
        """
        Record metrics for an optimization.
        """
        with self._lock:
            # Get or create metrics
            if optimization not in self.current_metrics:
                self.current_metrics[optimization] = OptimizationMetrics(name=optimization)
                
            metrics = self.current_metrics[optimization]
            
            # Update performance metrics
            if latency_ms is not None:
                metrics.latency_ms = latency_ms
                self._update_latency_percentiles(optimization, latency_ms)
                
            if throughput is not None:
                metrics.throughput = throughput
                
            if accuracy is not None:
                metrics.token_accuracy = accuracy
                
            # Update error counts
            metrics.total_requests += 1
            if error:
                metrics.error_count += 1
            if crash:
                metrics.crash_count += 1
            if timeout:
                metrics.timeout_count += 1
                
            # Update resource metrics
            self._update_resource_metrics(metrics)
            
            # Calculate comparative metrics
            self._calculate_comparative_metrics(optimization, metrics)
            
            # Add to history
            self.metric_history[optimization].append(metrics)
            
            # Check thresholds
            violations = self._check_thresholds(optimization, metrics)
            if violations:
                self._handle_violations(optimization, violations)
                
            self.total_measurements += 1
            
    def record_baseline(self, optimization: str, metrics: OptimizationMetrics):
        """
        Record baseline metrics for comparison.
        """
        with self._lock:
            self.baseline_metrics[optimization] = metrics
            logger.info(f"Baseline recorded for {optimization}")
            
    def get_health_status(self, optimization: str) -> Tuple[str, List[str]]:
        """
        Get health status of an optimization.
        Returns (status, issues)
        """
        with self._lock:
            if optimization not in self.current_metrics:
                return "NO_DATA", []
                
            metrics = self.current_metrics[optimization]
            violations = self._check_thresholds(optimization, metrics)
            
            if not violations:
                return "HEALTHY", []
            elif len(violations) <= 2:
                return "DEGRADED", violations
            else:
                return "CRITICAL", violations
                
    def get_metrics_summary(self, optimization: str) -> Dict[str, Any]:
        """
        Get summary of metrics for an optimization.
        """
        with self._lock:
            if optimization not in self.metric_history:
                return {}
                
            history = list(self.metric_history[optimization])
            if not history:
                return {}
                
            # Calculate aggregates
            latencies = [m.latency_ms for m in history if m.latency_ms > 0]
            accuracies = [m.token_accuracy for m in history if m.token_accuracy > 0]
            
            error_rate = sum(m.error_count for m in history) / max(1, sum(m.total_requests for m in history))
            
            return {
                'optimization': optimization,
                'sample_count': len(history),
                'latency': {
                    'mean': np.mean(latencies) if latencies else 0,
                    'p50': np.percentile(latencies, 50) if latencies else 0,
                    'p95': np.percentile(latencies, 95) if latencies else 0,
                    'p99': np.percentile(latencies, 99) if latencies else 0,
                },
                'accuracy': {
                    'mean': np.mean(accuracies) if accuracies else 1.0,
                    'min': min(accuracies) if accuracies else 1.0,
                },
                'error_rate': error_rate,
                'health': self.get_health_status(optimization)[0],
                'latest_metrics': self.current_metrics.get(optimization),
            }
            
    def _update_latency_percentiles(self, optimization: str, latency: float):
        """
        Update latency percentiles.
        """
        history = self.metric_history[optimization]
        latencies = [m.latency_ms for m in history if m.latency_ms > 0]
        latencies.append(latency)
        
        if len(latencies) >= 10:  # Need enough samples
            metrics = self.current_metrics[optimization]
            metrics.latency_p50 = np.percentile(latencies, 50)
            metrics.latency_p95 = np.percentile(latencies, 95)
            metrics.latency_p99 = np.percentile(latencies, 99)
            
    def _update_resource_metrics(self, metrics: OptimizationMetrics):
        """
        Update resource usage metrics.
        """
        # CPU usage
        metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        if torch.cuda.is_available():
            metrics.memory_gb = torch.cuda.memory_allocated() / 1e9
            
            # GPU utilization
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    metrics.gpu_utilization = gpus[0].load * 100
            except:
                pass
        else:
            metrics.memory_gb = psutil.Process().memory_info().rss / 1e9
            
    def _calculate_comparative_metrics(self, optimization: str, metrics: OptimizationMetrics):
        """
        Calculate metrics relative to baseline.
        """
        if optimization not in self.baseline_metrics:
            return
            
        baseline = self.baseline_metrics[optimization]
        
        # Ratios (1.0 = same as baseline)
        if baseline.latency_ms > 0:
            metrics.latency_ratio = metrics.latency_ms / baseline.latency_ms
            
        if baseline.throughput > 0:
            metrics.throughput_ratio = metrics.throughput / baseline.throughput
            
        if baseline.perplexity > 0:
            metrics.perplexity_ratio = metrics.perplexity / baseline.perplexity
            
        # Deltas
        metrics.accuracy_delta = metrics.token_accuracy - baseline.token_accuracy
        
    def _check_thresholds(self, optimization: str, metrics: OptimizationMetrics) -> List[str]:
        """
        Check if metrics violate thresholds.
        Returns list of violations.
        """
        violations = []
        
        # Performance checks
        if metrics.latency_ms > self.thresholds.max_latency_ms:
            violations.append(f"Latency {metrics.latency_ms:.1f}ms > {self.thresholds.max_latency_ms}ms")
            
        if metrics.latency_ratio > self.thresholds.max_latency_increase:
            violations.append(f"Latency increased {metrics.latency_ratio:.1%}")
            
        # Quality checks
        if metrics.token_accuracy < self.thresholds.min_token_accuracy:
            violations.append(f"Accuracy {metrics.token_accuracy:.1%} < {self.thresholds.min_token_accuracy:.1%}")
            
        if metrics.perplexity_ratio > self.thresholds.max_perplexity_increase:
            violations.append(f"Perplexity increased {metrics.perplexity_ratio:.1%}")
            
        # Stability checks
        if metrics.total_requests > 0:
            error_rate = metrics.error_count / metrics.total_requests
            if error_rate > self.thresholds.max_error_rate:
                violations.append(f"Error rate {error_rate:.1%} > {self.thresholds.max_error_rate:.1%}")
                
        # Resource checks
        if metrics.memory_gb > self.thresholds.max_memory_gb:
            violations.append(f"Memory {metrics.memory_gb:.1f}GB > {self.thresholds.max_memory_gb}GB")
            
        if metrics.cpu_percent > self.thresholds.max_cpu_percent:
            violations.append(f"CPU {metrics.cpu_percent:.1f}% > {self.thresholds.max_cpu_percent}%")
            
        return violations
        
    def _handle_violations(self, optimization: str, violations: List[str]):
        """
        Handle threshold violations.
        """
        # Track breach count
        self.breach_counts[optimization] += 1
        
        # Log violations
        logger.warning(f"{optimization} violations: {violations}")
        
        # Create alert
        alert = {
            'timestamp': datetime.now().isoformat(),
            'optimization': optimization,
            'violations': violations,
            'breach_count': self.breach_counts[optimization],
            'metrics': self.current_metrics[optimization]
        }
        
        self.alert_history.append(alert)
        
        # Trigger callback if provided
        if self.alert_callback:
            self.alert_callback(optimization, violations, alert)
            
    def check_memory_leak(self, optimization: str, hours: float = 1.0) -> bool:
        """
        Check for memory leaks over time window.
        """
        with self._lock:
            history = list(self.metric_history[optimization])
            if len(history) < 2:
                return False
                
            # Get metrics from past hour
            cutoff_time = time.time() - (hours * 3600)
            recent_metrics = [m for m in history if m.timestamp > cutoff_time]
            
            if len(recent_metrics) < 2:
                return False
                
            # Calculate memory growth rate
            start_memory = recent_metrics[0].memory_gb
            end_memory = recent_metrics[-1].memory_gb
            time_diff = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp) / 3600  # hours
            
            if time_diff > 0:
                growth_rate = (end_memory - start_memory) / time_diff
                return growth_rate > self.thresholds.max_memory_growth_gb_per_hour
                
        return False
        
    def generate_report(self, optimization: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate health report for optimization(s).
        """
        with self._lock:
            if optimization:
                optimizations = [optimization]
            else:
                optimizations = list(self.current_metrics.keys())
                
            report = {
                'timestamp': datetime.now().isoformat(),
                'monitor_uptime': time.time() - self.start_time,
                'total_measurements': self.total_measurements,
                'optimizations': {}
            }
            
            for opt in optimizations:
                summary = self.get_metrics_summary(opt)
                health_status, issues = self.get_health_status(opt)
                
                report['optimizations'][opt] = {
                    'health': health_status,
                    'issues': issues,
                    'breach_count': self.breach_counts[opt],
                    'memory_leak': self.check_memory_leak(opt),
                    'summary': summary
                }
                
            # Add recent alerts
            report['recent_alerts'] = self.alert_history[-10:]  # Last 10 alerts
            
            return report
            
    def export_metrics(self, filepath: str):
        """
        Export metrics to file for analysis.
        """
        with self._lock:
            data = {
                'export_time': datetime.now().isoformat(),
                'metrics': {},
                'baselines': {},
                'thresholds': self.thresholds.__dict__,
                'alerts': self.alert_history
            }
            
            # Export all metrics
            for opt, history in self.metric_history.items():
                data['metrics'][opt] = [
                    {k: v for k, v in m.__dict__.items()}
                    for m in history
                ]
                
            # Export baselines
            for opt, baseline in self.baseline_metrics.items():
                data['baselines'][opt] = {k: v for k, v in baseline.__dict__.items()}
                
            # Save to file
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            logger.info(f"Metrics exported to {filepath}")
            

class MetricsCollector:
    """
    Helper class to collect metrics from model inference.
    """
    
    def __init__(self, monitor: OptimizationHealthMonitor, optimization: str):
        self.monitor = monitor
        self.optimization = optimization
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate latency
        latency_ms = (time.time() - self.start_time) * 1000
        
        # Determine if error occurred
        error = exc_type is not None
        crash = exc_type is not None and not isinstance(exc_val, (TimeoutError, ValueError))
        timeout = isinstance(exc_val, TimeoutError)
        
        # Record metrics
        self.monitor.record_metrics(
            self.optimization,
            latency_ms=latency_ms,
            error=error,
            crash=crash,
            timeout=timeout
        )
        
        # Don't suppress exceptions
        return False


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    def alert_handler(opt, violations, alert):
        print(f"ALERT for {opt}: {violations}")
        
    monitor = OptimizationHealthMonitor(alert_callback=alert_handler)
    
    # Simulate baseline
    baseline = OptimizationMetrics(
        name="test_optimization",
        latency_ms=100,
        throughput=100,
        token_accuracy=0.99
    )
    monitor.record_baseline("test_optimization", baseline)
    
    # Simulate metrics collection
    for i in range(50):
        # Simulate degrading performance
        latency = 100 + i * 2  # Increasing latency
        accuracy = 0.99 - i * 0.001  # Decreasing accuracy
        
        monitor.record_metrics(
            "test_optimization",
            latency_ms=latency,
            accuracy=accuracy,
            error=(i % 20 == 0)  # Occasional errors
        )
        
        time.sleep(0.1)
        
    # Generate report
    report = monitor.generate_report()
    print("\nHealth Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Export metrics
    monitor.export_metrics("test_metrics.json")
    print("\nMetrics exported.")