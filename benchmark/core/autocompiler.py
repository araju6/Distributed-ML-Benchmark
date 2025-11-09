"""
AutoCompiler - Test Mode

Runs benchmarks across all available compilers in parallel and returns
comparison statistics to help users choose the best compiler for their model.
"""

import torch
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime

from .benchmark_runner import BenchmarkRunner
from .metrics import BenchmarkMetrics
from ..compilers.base import Compiler
from ..models.base import ModelWrapper
from ..utils.device import get_device
from ..compilers.pytorch_eager import PyTorchEagerCompiler
from ..compilers.torchscript import TorchScriptCompiler
from ..compilers.torch_inductor import TorchInductorCompiler
from ..compilers.onnx_runtime import ONNXRuntimeCompiler
from ..compilers.tvm import TVMCompiler
from ..compilers.tensorrt import TensorRTCompiler


@dataclass
class CompilerResult:
    """Result for a single compiler benchmark."""
    compiler_name: str
    compiler_display_name: str
    success: bool
    error_message: Optional[str] = None
    metrics: Optional[BenchmarkMetrics] = None
    compile_time: Optional[float] = None


@dataclass
class AutoCompilerReport:
    """Complete report from AutoCompiler test."""
    model_name: str
    batch_size: int
    input_format: Dict[str, Any]  # input_shape or max_length
    timestamp: str
    results: List[CompilerResult]
    recommendations: Dict[str, Any]
    summary: Dict[str, Any]


class AutoCompiler:
    """
    AutoCompiler - Test Mode
    
    Runs benchmarks across all available compilers in parallel and generates
    a comparison report to help users choose the best compiler.
    """
    
    # List of all available compilers to test
    # Note: Some compilers may fail on certain hardware (e.g., torch_inductor on P100)
    AVAILABLE_COMPILERS = [
        ("pytorch_eager", "PyTorch Eager", lambda: PyTorchEagerCompiler()),
        ("torchscript_trace", "TorchScript (Trace)", lambda: TorchScriptCompiler(method="trace")),
        ("torchscript_script", "TorchScript (Script)", lambda: TorchScriptCompiler(method="script")),
        ("torch_inductor", "TorchInductor", lambda: TorchInductorCompiler(mode="default")),
        ("onnx_runtime", "ONNX Runtime", lambda: ONNXRuntimeCompiler()),
        ("tvm", "TVM", lambda: TVMCompiler()),
        ("tvm_autotuned", "TVM (Autotuned)", lambda: TVMCompiler(use_autotuning=True)),
        ("tensorrt_fp32", "TensorRT (FP32)", lambda: TensorRTCompiler(fp16_mode=False)),
        ("tensorrt_fp16", "TensorRT (FP16)", lambda: TensorRTCompiler(fp16_mode=True)),
    ]
    
    def __init__(
        self,
        warmup_iters: int = 10,
        measured_iters: int = 50,  # Fewer iterations for quick testing
        use_ray: bool = True,
        device: Optional[torch.device] = None
    ):
        """Initialize AutoCompiler.
        
        Args:
            warmup_iters: Number of warmup iterations
            measured_iters: Number of measured iterations (reduced for quick testing)
            use_ray: Whether to use Ray for parallel execution
            device: Device to use (auto-detect if None)
        """
        self.warmup_iters = warmup_iters
        self.measured_iters = measured_iters
        self.use_ray = use_ray
        self.device = device or get_device()
        
        # Try to initialize Ray if requested
        if self.use_ray:
            try:
                import ray
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                self.ray_available = True
            except Exception as e:
                print(f"Warning: Ray not available, falling back to sequential execution: {e}")
                self.ray_available = False
                self.use_ray = False
        else:
            self.ray_available = False
    
    def cleanup(self):
        """Clean up resources (shutdown Ray if initialized)."""
        if self.ray_available and self.use_ray:
            try:
                import ray
                if ray.is_initialized():
                    ray.shutdown()
            except Exception:
                pass  # Ignore errors during cleanup
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
    
    def get_available_compilers(self) -> List[Tuple[str, str]]:
        """Get list of available compiler names and display names."""
        return [(name, display) for name, display, _ in self.AVAILABLE_COMPILERS]
    
    def test_compiler(
        self,
        model_wrapper: ModelWrapper,
        compiler_name: str,
        compiler_factory,
        batch_size: int
    ) -> CompilerResult:
        """Test a single compiler.
        
        Args:
            model_wrapper: Model wrapper to test
            compiler_name: Name of the compiler
            compiler_factory: Function to create compiler instance
            batch_size: Batch size for testing
        
        Returns:
            CompilerResult with metrics or error
        """
        display_name = next(
            (display for name, display, _ in self.AVAILABLE_COMPILERS if name == compiler_name),
            compiler_name
        )
        
        try:
            # Create compiler
            compiler = compiler_factory()
            
            # Run benchmark
            runner = BenchmarkRunner(
                device=self.device,
                warmup_iters=self.warmup_iters,
                measured_iters=self.measured_iters
            )
            
            metrics = runner.run_benchmark(model_wrapper, compiler, batch_size)
            
            return CompilerResult(
                compiler_name=compiler_name,
                compiler_display_name=display_name,
                success=True,
                metrics=metrics,
                compile_time=metrics.compile_time if hasattr(metrics, 'compile_time') else None
            )
            
        except Exception as e:
            return CompilerResult(
                compiler_name=compiler_name,
                compiler_display_name=display_name,
                success=False,
                error_message=str(e)
            )
    
    def test_all_compilers(
        self,
        model_wrapper: ModelWrapper,
        batch_size: int,
        compiler_filter: Optional[List[str]] = None
    ) -> List[CompilerResult]:
        """Test all available compilers (or filtered subset).
        
        Args:
            model_wrapper: Model wrapper to test
            batch_size: Batch size for testing
            compiler_filter: Optional list of compiler names to test (None = all)
        
        Returns:
            List of CompilerResult objects
        """
        # Filter compilers if requested
        compilers_to_test = self.AVAILABLE_COMPILERS
        if compiler_filter:
            compilers_to_test = [
                (name, display, factory) for name, display, factory in compilers_to_test
                if name in compiler_filter
            ]
        
        print(f"\n{'='*70}")
        print(f"AutoCompiler Test Mode")
        print(f"{'='*70}")
        print(f"Model: {model_wrapper.get_name()}")
        print(f"Batch Size: {batch_size}")
        print(f"Testing {len(compilers_to_test)} compilers...")
        print(f"Execution: {'Parallel (Ray)' if self.use_ray and self.ray_available else 'Sequential'}")
        print(f"{'='*70}\n")
        
        if self.use_ray and self.ray_available:
            return self._test_parallel_ray(model_wrapper, batch_size, compilers_to_test)
        else:
            return self._test_sequential(model_wrapper, batch_size, compilers_to_test)
    
    def _test_sequential(
        self,
        model_wrapper: ModelWrapper,
        batch_size: int,
        compilers_to_test: List[Tuple[str, str, Any]]
    ) -> List[CompilerResult]:
        """Test compilers sequentially."""
        results = []
        
        for i, (compiler_name, display_name, compiler_factory) in enumerate(compilers_to_test, 1):
            print(f"\n[{i}/{len(compilers_to_test)}] Testing {display_name}...")
            result = self.test_compiler(model_wrapper, compiler_name, compiler_factory, batch_size)
            results.append(result)
            
            if result.success:
                print(f"  ✓ Success: Latency={result.metrics.latency_mean:.3f}ms, "
                  f"Throughput={result.metrics.throughput:.2f} samples/sec")
            else:
                print(f"  ✗ Failed: {result.error_message}")
        
        return results
    
    def _test_parallel_ray(
        self,
        model_wrapper: ModelWrapper,
        batch_size: int,
        compilers_to_test: List[Tuple[str, str, Any]]
    ) -> List[CompilerResult]:
        """Test compilers in parallel using Ray."""
        import ray
        
        # Create a remote function for testing
        @ray.remote(num_gpus=1)
        def test_compiler_remote(
            model_name: str,
            model_config: dict,
            compiler_name: str,
            batch_size: int,
            warmup_iters: int,
            measured_iters: int
        ) -> Dict[str, Any]:
            """Ray remote function to test a compiler."""
            from benchmark.models.resnet import ResNetWrapper
            from benchmark.models.mobilenet import MobileNetWrapper
            from benchmark.models.bert import BERTWrapper
            from benchmark.models.gpt2 import GPT2Wrapper
            from benchmark.utils.device import get_device
            from benchmark.core.benchmark_runner import BenchmarkRunner
            from benchmark.compilers.pytorch_eager import PyTorchEagerCompiler
            from benchmark.compilers.torchscript import TorchScriptCompiler
            from benchmark.compilers.torch_inductor import TorchInductorCompiler
            from benchmark.compilers.onnx_runtime import ONNXRuntimeCompiler
            from benchmark.compilers.tvm import TVMCompiler
            from benchmark.compilers.tensorrt import TensorRTCompiler
            
            device = get_device()
            
            # Recreate model wrapper
            if model_name == "resnet50":
                model_wrapper = ResNetWrapper(input_shape=tuple(model_config['input_shape']), pretrained=True)
            elif model_name == "mobilenet_v3_large":
                model_wrapper = MobileNetWrapper(input_shape=tuple(model_config['input_shape']), pretrained=True)
            elif model_name == "bert_base":
                model_wrapper = BERTWrapper(max_length=model_config['max_length'], pretrained=True)
            elif model_name == "gpt2":
                model_wrapper = GPT2Wrapper(max_length=model_config['max_length'], pretrained=True)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Create compiler (recreate factory functions locally)
            compiler_map = {
                "pytorch_eager": lambda: PyTorchEagerCompiler(),
                "torchscript_trace": lambda: TorchScriptCompiler(method="trace"),
                "torchscript_script": lambda: TorchScriptCompiler(method="script"),
                "torch_inductor": lambda: TorchInductorCompiler(mode="default"),
                "onnx_runtime": lambda: ONNXRuntimeCompiler(),
                "tvm": lambda: TVMCompiler(),
                "tvm_autotuned": lambda: TVMCompiler(use_autotuning=True),
                "tensorrt_fp32": lambda: TensorRTCompiler(fp16_mode=False),
                "tensorrt_fp16": lambda: TensorRTCompiler(fp16_mode=True),
            }
            
            if compiler_name not in compiler_map:
                raise ValueError(f"Unknown compiler: {compiler_name}")
            
            compiler = compiler_map[compiler_name]()
            
            # Run benchmark
            runner = BenchmarkRunner(
                device=device,
                warmup_iters=warmup_iters,
                measured_iters=measured_iters
            )
            
            try:
                metrics = runner.run_benchmark(model_wrapper, compiler, batch_size)
                return {
                    "success": True,
                    "metrics": asdict(metrics),
                    "compile_time": metrics.compile_time if hasattr(metrics, 'compile_time') else None
                }
            except Exception as e:
                return {
                    "success": False,
                    "error_message": str(e)
                }
        
        # Prepare model config for serialization
        model_name = model_wrapper.get_name()
        # Extract base model name (e.g., "resnet50_224x224" -> "resnet50")
        base_model_name = model_name.split('_')[0]
        
        model_config = {}
        if hasattr(model_wrapper, 'input_shape'):
            model_config['input_shape'] = list(model_wrapper.input_shape)
        if hasattr(model_wrapper, 'max_length'):
            model_config['max_length'] = model_wrapper.max_length
        
        # Submit all tasks
        tasks = []
        for compiler_name, display_name, _ in compilers_to_test:
            print(f"Submitting test for {display_name}...")
            task = test_compiler_remote.remote(
                base_model_name,
                model_config,
                compiler_name,
                batch_size,
                self.warmup_iters,
                self.measured_iters
            )
            tasks.append((compiler_name, display_name, task))
        
        # Collect results
        results = []
        for i, (compiler_name, display_name, task) in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] Waiting for {display_name}...")
            try:
                result_dict = ray.get(task)
                
                if result_dict["success"]:
                    # Reconstruct metrics from dict
                    metrics = BenchmarkMetrics(**result_dict["metrics"])
                    result = CompilerResult(
                        compiler_name=compiler_name,
                        compiler_display_name=display_name,
                        success=True,
                        metrics=metrics,
                        compile_time=result_dict.get("compile_time")
                    )
                    print(f"  ✓ Success: Latency={metrics.latency_mean:.3f}ms, "
                          f"Throughput={metrics.throughput:.2f} samples/sec")
                else:
                    result = CompilerResult(
                        compiler_name=compiler_name,
                        compiler_display_name=display_name,
                        success=False,
                        error_message=result_dict.get("error_message", "Unknown error")
                    )
                    print(f"  ✗ Failed: {result.error_message}")
                
                results.append(result)
            except Exception as e:
                result = CompilerResult(
                    compiler_name=compiler_name,
                    compiler_display_name=display_name,
                    success=False,
                    error_message=str(e)
                )
                results.append(result)
                print(f"  ✗ Failed: {e}")
        
        return results
    
    def generate_report(
        self,
        model_wrapper: ModelWrapper,
        batch_size: int,
        results: List[CompilerResult]
    ) -> AutoCompilerReport:
        """Generate comparison report from results.
        
        Args:
            model_wrapper: Model wrapper that was tested
            batch_size: Batch size used
            results: List of compiler results
        
        Returns:
            AutoCompilerReport with recommendations
        """
        # Filter successful results
        successful_results = [r for r in results if r.success and r.metrics]
        
        if not successful_results:
            return AutoCompilerReport(
                model_name=model_wrapper.get_name(),
                batch_size=batch_size,
                input_format={},
                timestamp=datetime.now().isoformat(),
                results=results,
                recommendations={"error": "No compilers succeeded"},
                summary={"total_tested": len(results), "successful": 0, "failed": len(results)}
            )
        
        # Find best performers
        best_latency = min(successful_results, key=lambda r: r.metrics.latency_mean)
        best_throughput = max(successful_results, key=lambda r: r.metrics.throughput)
        best_memory = min(successful_results, key=lambda r: r.metrics.peak_memory_mb)
        
        # Get input format
        input_format = {}
        if hasattr(model_wrapper, 'input_shape'):
            input_format['input_shape'] = list(model_wrapper.input_shape)
        if hasattr(model_wrapper, 'max_length'):
            input_format['max_length'] = model_wrapper.max_length
        
        # Generate recommendations
        recommendations = {
            "best_latency": {
                "compiler": best_latency.compiler_name,
                "display_name": best_latency.compiler_display_name,
                "latency_ms": best_latency.metrics.latency_mean,
                "throughput": best_latency.metrics.throughput
            },
            "best_throughput": {
                "compiler": best_throughput.compiler_name,
                "display_name": best_throughput.compiler_display_name,
                "latency_ms": best_throughput.metrics.latency_mean,
                "throughput": best_throughput.metrics.throughput
            },
            "best_memory": {
                "compiler": best_memory.compiler_name,
                "display_name": best_memory.compiler_display_name,
                "peak_memory_mb": best_memory.metrics.peak_memory_mb,
                "latency_ms": best_memory.metrics.latency_mean
            },
            "balanced": {
                "compiler": best_throughput.compiler_name,  # Usually throughput correlates with good latency
                "display_name": best_throughput.compiler_display_name,
                "reason": "Best overall throughput with reasonable latency"
            }
        }
        
        # Summary statistics
        summary = {
            "total_tested": len(results),
            "successful": len(successful_results),
            "failed": len(results) - len(successful_results),
            "avg_latency_ms": sum(r.metrics.latency_mean for r in successful_results) / len(successful_results),
            "max_throughput": best_throughput.metrics.throughput,
            "min_memory_mb": best_memory.metrics.peak_memory_mb
        }
        
        return AutoCompilerReport(
            model_name=model_wrapper.get_name(),
            batch_size=batch_size,
            input_format=input_format,
            timestamp=datetime.now().isoformat(),
            results=results,
            recommendations=recommendations,
            summary=summary
        )
    
    def print_report(self, report: AutoCompilerReport):
        """Print a formatted report to console."""
        print(f"\n{'='*70}")
        print(f"AUTOCOMPILER TEST REPORT")
        print(f"{'='*70}")
        print(f"Model: {report.model_name}")
        print(f"Batch Size: {report.batch_size}")
        print(f"Timestamp: {report.timestamp}")
        print(f"\nSummary:")
        print(f"  Total Compilers Tested: {report.summary['total_tested']}")
        print(f"  Successful: {report.summary['successful']}")
        print(f"  Failed: {report.summary['failed']}")
        
        if report.summary['successful'] > 0:
            print(f"  Average Latency: {report.summary['avg_latency_ms']:.3f} ms")
            print(f"  Max Throughput: {report.summary['max_throughput']:.2f} samples/sec")
            print(f"  Min Memory: {report.summary['min_memory_mb']:.2f} MB")
        
        print(f"\n{'='*70}")
        print(f"RECOMMENDATIONS")
        print(f"{'='*70}")
        
        if "error" in report.recommendations:
            print(f"Error: {report.recommendations['error']}")
        else:
            print(f"\n🏆 Best Latency:")
            rec = report.recommendations['best_latency']
            print(f"   Compiler: {rec['display_name']}")
            print(f"   Latency: {rec['latency_ms']:.3f} ms")
            print(f"   Throughput: {rec['throughput']:.2f} samples/sec")
            
            print(f"\n🚀 Best Throughput:")
            rec = report.recommendations['best_throughput']
            print(f"   Compiler: {rec['display_name']}")
            print(f"   Latency: {rec['latency_ms']:.3f} ms")
            print(f"   Throughput: {rec['throughput']:.2f} samples/sec")
            
            print(f"\n💾 Best Memory Efficiency:")
            rec = report.recommendations['best_memory']
            print(f"   Compiler: {rec['display_name']}")
            print(f"   Peak Memory: {rec['peak_memory_mb']:.2f} MB")
            print(f"   Latency: {rec['latency_ms']:.3f} ms")
            
            print(f"\n⚖️  Balanced Recommendation:")
            rec = report.recommendations['balanced']
            print(f"   Compiler: {rec['display_name']}")
            print(f"   Reason: {rec['reason']}")
        
        print(f"\n{'='*70}")
        print(f"DETAILED RESULTS")
        print(f"{'='*70}")
        
        for result in report.results:
            print(f"\n{result.compiler_display_name}:")
            if result.success and result.metrics:
                print(f"  Status: ✓ Success")
                print(f"  Latency (mean): {result.metrics.latency_mean:.3f} ms")
                print(f"  Latency (p95): {result.metrics.latency_p95:.3f} ms")
                print(f"  Throughput: {result.metrics.throughput:.2f} samples/sec")
                print(f"  Peak Memory: {result.metrics.peak_memory_mb:.2f} MB")
                if result.compile_time:
                    print(f"  Compile Time: {result.compile_time:.3f} s")
            else:
                print(f"  Status: ✗ Failed")
                print(f"  Error: {result.error_message}")
        
        print(f"\n{'='*70}\n")
    
    def save_report(self, report: AutoCompilerReport, output_path: str):
        """Save report to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict (handling nested dataclasses)
        report_dict = {
            "model_name": report.model_name,
            "batch_size": report.batch_size,
            "input_format": report.input_format,
            "timestamp": report.timestamp,
            "results": [
                {
                    "compiler_name": r.compiler_name,
                    "compiler_display_name": r.compiler_display_name,
                    "success": r.success,
                    "error_message": r.error_message,
                    "metrics": asdict(r.metrics) if r.metrics else None,
                    "compile_time": r.compile_time
                }
                for r in report.results
            ],
            "recommendations": report.recommendations,
            "summary": report.summary
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"Report saved to: {output_file}")

