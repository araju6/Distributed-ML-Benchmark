from prometheus_client import Counter, Histogram, Gauge, start_http_server
from typing import Optional
import os

# Prometheus metrics definitions
benchmark_latency_seconds = Histogram(
    'benchmark_latency_seconds',
    'Benchmark inference latency in seconds',
    ['compiler', 'model', 'batch_size', 'gpu_id'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

benchmark_throughput_samples_per_sec = Gauge(
    'benchmark_throughput_samples_per_sec',
    'Benchmark throughput in samples per second',
    ['compiler', 'model', 'batch_size', 'gpu_id']
)

benchmark_gpu_memory_mb = Gauge(
    'benchmark_gpu_memory_mb',
    'GPU memory usage in MB',
    ['gpu_id', 'type']  # type: 'peak' or 'avg'
)

benchmark_compile_time_seconds = Histogram(
    'benchmark_compile_time_seconds',
    'Model compilation time in seconds',
    ['compiler', 'model'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

benchmark_runs_total = Counter(
    'benchmark_runs_total',
    'Total number of benchmark runs completed',
    ['compiler', 'model', 'batch_size', 'status']  # status: 'success' or 'failure'
)

benchmark_iterations_total = Counter(
    'benchmark_iterations_total',
    'Total number of benchmark iterations executed',
    ['compiler', 'model', 'batch_size']
)


class PrometheusMetricsExporter:
    """Manager for Prometheus metrics export."""
    
    def __init__(self, port: int = 8000, enabled: bool = True):
        """Initialize Prometheus metrics exporter.
        
        Args:
            port: HTTP port to expose metrics endpoint
            enabled: Whether to enable metrics export
        """
        self.enabled = enabled
        self.port = port
        self.server_started = False
        
        if enabled:
            self.start_server()
    
    def start_server(self):
        """Start HTTP server for Prometheus to scrape metrics."""
        if not self.server_started:
            try:
                start_http_server(self.port)
                print(f"Prometheus metrics server started on port {self.port}")
                print(f"Metrics available at http://localhost:{self.port}/metrics")
                self.server_started = True
            except OSError as e:
                print(f"Warning: Could not start Prometheus metrics server on port {self.port}: {e}")
                print("Metrics collection will continue but won't be exposed via HTTP")
                self.enabled = False
    
    def record_latency(self, latency_seconds: float, compiler: str, model: str, batch_size: int, gpu_id: int = 0):
        """Record inference latency."""
        if self.enabled:
            benchmark_latency_seconds.labels(
                compiler=compiler,
                model=model,
                batch_size=str(batch_size),
                gpu_id=str(gpu_id)
            ).observe(latency_seconds)
    
    def record_throughput(self, throughput: float, compiler: str, model: str, batch_size: int, gpu_id: int = 0):
        """Record throughput."""
        if self.enabled:
            benchmark_throughput_samples_per_sec.labels(
                compiler=compiler,
                model=model,
                batch_size=str(batch_size),
                gpu_id=str(gpu_id)
            ).set(throughput)
    
    def record_gpu_memory(self, memory_mb: float, gpu_id: int = 0, memory_type: str = 'peak'):
        """Record GPU memory usage."""
        if self.enabled:
            benchmark_gpu_memory_mb.labels(
                gpu_id=str(gpu_id),
                type=memory_type
            ).set(memory_mb)
    
    def record_compile_time(self, compile_time_seconds: float, compiler: str, model: str):
        """Record compilation time."""
        if self.enabled:
            benchmark_compile_time_seconds.labels(
                compiler=compiler,
                model=model
            ).observe(compile_time_seconds)
    
    def record_benchmark_run(self, compiler: str, model: str, batch_size: int, status: str = 'success'):
        """Record a completed benchmark run."""
        if self.enabled:
            benchmark_runs_total.labels(
                compiler=compiler,
                model=model,
                batch_size=str(batch_size),
                status=status
            ).inc()
    
    def record_iterations(self, num_iterations: int, compiler: str, model: str, batch_size: int):
        """Record number of iterations executed."""
        if self.enabled:
            benchmark_iterations_total.labels(
                compiler=compiler,
                model=model,
                batch_size=str(batch_size)
            ).inc(num_iterations)

