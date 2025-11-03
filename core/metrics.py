import torch
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkMetrics:
    """container for benchmark results"""
    compiler_name: str
    model_name: str
    batch_size: int
    
    latency_mean: float
    latency_std: float
    latency_p50: float
    latency_p95: float
    
    throughput: float
    
    peak_memory_mb: float
    avg_memory_mb: float
    
    compile_time_sec: float = None
    
    def to_dict(self):
        return {
            'compiler': self.compiler_name,
            'model': self.model_name,
            'batch_size': self.batch_size,
            'latency_mean_ms': f"{self.latency_mean:.3f}",
            'latency_std_ms': f"{self.latency_std:.3f}",
            'latency_p50_ms': f"{self.latency_p50:.3f}",
            'latency_p95_ms': f"{self.latency_p95:.3f}",
            'throughput_samples_per_sec': f"{self.throughput:.2f}",
            'peak_memory_mb': f"{self.peak_memory_mb:.2f}",
            'avg_memory_mb': f"{self.avg_memory_mb:.2f}",
            'compile_time_sec': f"{self.compile_time_sec:.3f}" if self.compile_time_sec else "N/A"
        }


class MetricsCollector:
    @staticmethod
    def compute_metrics(latencies: List[float], memory_readings: List[float], batch_size: int, compile_time: float = None) -> dict:

        latencies_ms = np.array(latencies) * 1000
        memory_mb = np.array(memory_readings) / (1024 ** 2)
        
        latency_mean = float(np.mean(latencies_ms))
        latency_std = float(np.std(latencies_ms))
        latency_p50 = float(np.percentile(latencies_ms, 50))
        latency_p95 = float(np.percentile(latencies_ms, 95))
        
        avg_latency_sec = np.mean(latencies)
        throughput = batch_size / avg_latency_sec
        
        peak_memory = float(np.max(memory_mb))
        avg_memory = float(np.mean(memory_mb))
        
        return {
            'latency_mean': latency_mean,
            'latency_std': latency_std,
            'latency_p50': latency_p50,
            'latency_p95': latency_p95,
            'throughput': throughput,
            'peak_memory_mb': peak_memory,
            'avg_memory_mb': avg_memory,
            'compile_time_sec': compile_time
        }