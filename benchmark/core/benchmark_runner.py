import time
import torch
import torch.nn as nn
from typing import List, Optional
from ..compilers.base import Compiler
from ..models.base import ModelWrapper
from ..utils.device import GPUMonitor
from .metrics import MetricsCollector, BenchmarkMetrics
from .nsight_profiler import NsightProfiler

class BenchmarkRunner:
    
    def __init__(
        self,
        device: torch.device,
        warmup_iters: int,
        measured_iters: int,
        nsight_profiler: Optional[NsightProfiler] = None
    ):
        self.device = device
        self.warmup_iters = warmup_iters
        self.measured_iters = measured_iters
        self.gpu_monitor = GPUMonitor(device)
        self.nsight_profiler = nsight_profiler
    
    def run_benchmark(self, model_wrapper: ModelWrapper, compiler: Compiler, batch_size: int) -> BenchmarkMetrics:
        """Run a full benchmark pass for one model+compiler+batch.

        Does a compile step, a short warmup, and then measures latency and
        memory over several iterations. Returns a small dataclass with the
        aggregated stats so callers can save/compare.
        """

        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_wrapper.get_name()} | {compiler.get_name()} | batch_size={batch_size}")
        print(f"{'='*60}")
        
        model = model_wrapper.get_model().to(self.device)
        example_input = model_wrapper.get_example_input(batch_size, self.device)
        
        print("Compiling model...")
        compile_start_time = time.time()
        compiled_model = compiler.compile(model, example_input)
        
        with torch.no_grad():
            _ = compiled_model(example_input)
            self.gpu_monitor.synchronize()
        
        compile_time = time.time() - compile_start_time
        print(f"Compilation time: {compile_time:.3f}s")
        
        self.gpu_monitor.reset_peak_memory()
        
        print(f"Warming up ({self.warmup_iters} iterations)...")
        with torch.no_grad():
            for _ in range(self.warmup_iters):
                _ = compiled_model(example_input)
                self.gpu_monitor.synchronize()
        
        self.gpu_monitor.reset_peak_memory()
        
        print(f"Measuring ({self.measured_iters} iterations)...")
        iter_latencies = []
        
        with torch.no_grad():
            for i in range(self.measured_iters):
                t0 = time.time()
                _ = compiled_model(example_input)
                self.gpu_monitor.synchronize()
                latency = time.time() - t0
                iter_latencies.append(latency)
                
                if (i + 1) % 25 == 0:
                    print(f"  Progress: {i+1}/{self.measured_iters}")
        
        # get peak memory after all iterations (we reset before measurement)
        peak_mem_bytes = self.gpu_monitor.get_peak_memory()
        # avg memory is harder to track accurately without per-iteration sampling
        # using peak as approximation (memory is usually stable during inference)
        calc_stats = MetricsCollector.compute_metrics(
            latencies=iter_latencies,
            memory_readings=[peak_mem_bytes],  # single value, both peak and avg will use it
            batch_size=batch_size,
            compile_time=compile_time if compiler.get_name() != "pytorch_eager" and compile_time > 0 else None
        )
        
        metrics = BenchmarkMetrics(
            compiler_name=compiler.get_name(),
            model_name=model_wrapper.get_name(),
            batch_size=batch_size,
            **calc_stats
        )
        
        print(f"\nResults:")
        print(f"  Latency (mean): {metrics.latency_mean:.3f} ms")
        print(f"  Latency (p95): {metrics.latency_p95:.3f} ms")
        print(f"  Throughput: {metrics.throughput:.2f} samples/sec")
        print(f"  Peak Memory: {metrics.peak_memory_mb:.2f} MB")
        print(f"  Avg Memory: {metrics.avg_memory_mb:.2f} MB")
        
        # Run Nsight Systems profiling if enabled
        if self.nsight_profiler and self.nsight_profiler.is_available():
            self._run_nsight_profiling(
                compiled_model,
                example_input,
                model_wrapper.get_name(),
                compiler.get_name(),
                batch_size
            )
        
        del model, compiled_model, example_input
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return metrics
    
    def _run_nsight_profiling(
        self,
        compiled_model,
        example_input,
        model_name: str,
        compiler_name: str,
        batch_size: int
    ):
        """Run Nsight Systems profiling on a subset of iterations.
        
        This runs a separate profiling pass using nsys to profile CUDA kernels.
        Uses a subprocess approach to run the profiling script with nsys.
        """
        if not self.nsight_profiler or not self.nsight_profiler.is_available():
            return
        
        # Determine model config for profiling script
        model_config = {}
        if hasattr(example_input, 'shape'):
            if len(example_input.shape) == 4:  # Vision model: (batch, C, H, W)
                model_config['input_shape'] = list(example_input.shape[1:])
            elif len(example_input.shape) == 2:  # NLP model: (batch, seq_len)
                # For NLP models, use sequence length as max_length
                model_config['max_length'] = int(example_input.shape[1])
        
        profile_iterations = self.nsight_profiler.profile_iterations
        
        print(f"\nRunning Nsight Systems profiling ({profile_iterations} iterations)...")
        
        # Run profiling in subprocess
        profile_path = self.nsight_profiler.run_profiling_subprocess(
            model_name=model_name,
            compiler_name=compiler_name,
            batch_size=batch_size,
            model_config=model_config,
            num_iterations=profile_iterations
        )
        
        if profile_path:
            print(f"✓ Profile saved: {profile_path}")
            print(f"  View with: nsys-ui {profile_path}")
        else:
            print("  Profiling completed (check for errors above)")