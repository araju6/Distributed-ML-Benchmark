import os
import ray
from typing import List, Optional
from ..compilers.base import Compiler
from ..models.base import ModelWrapper
from ..core.benchmark_runner import BenchmarkRunner
from ..core.metrics import BenchmarkMetrics
from ..utils.device import get_device
from ..utils.ray_resources import setup_gpu_environment, get_gpu_assignments

@ray.remote(num_gpus=1)
def run_benchmark_task(
    model_name: str,
    compiler_name: str,
    batch_size: int,
    model_config: dict,  # Serialized model config
    warmup_iters: int,
    measured_iters: int,
    gpu_id: int,
    profiling_config: Optional[dict] = None,  # Profiling config dict (enabled, output_dir, profile_iterations)
    prometheus_config: Optional[dict] = None  # Prometheus config dict (enabled, port)
) -> BenchmarkMetrics:
    """Ray remote function to run a single benchmark task on a specific GPU.
    
    This function is executed in a separate Ray actor with GPU isolation.
    Each task gets one GPU via CUDA_VISIBLE_DEVICES.
    
    Args:
        model_name: Name of the model to benchmark
        compiler_name: Name of the compiler to use
        batch_size: Batch size for the benchmark
        model_config: Dictionary with model configuration (input_shape or max_length)
        warmup_iters: Number of warmup iterations
        measured_iters: Number of measured iterations
        gpu_id: GPU ID to use (will be isolated via CUDA_VISIBLE_DEVICES)
        profiling_config: Optional dict with profiling settings (enabled, output_dir, profile_iterations)
    
    Returns:
        BenchmarkMetrics from the benchmark run
    """
        
    setup_gpu_environment(gpu_id)
    
    # Import here to avoid circular dependencies and ensure imports happen after GPU setup
    from benchmark.core.benchmark_runner import BenchmarkRunner
    from benchmark.core.nsight_profiler import NsightProfiler
    from benchmark.core.prometheus_metrics import PrometheusMetricsExporter
    from benchmark.utils.device import get_device
    from benchmark.models.resnet import ResNetWrapper
    from benchmark.models.mobilenet import MobileNetWrapper
    from benchmark.models.bert import BERTWrapper
    from benchmark.models.gpt2 import GPT2Wrapper
    from benchmark.compilers.pytorch_eager import PyTorchEagerCompiler
    from benchmark.compilers.torch_inductor import TorchInductorCompiler
    from benchmark.compilers.torchscript import TorchScriptCompiler
    from benchmark.compilers.onnx_runtime import ONNXRuntimeCompiler
    from benchmark.compilers.tvm import TVMCompiler
    from benchmark.compilers.tensorrt import TensorRTCompiler
    
    # Get device (should be cuda:0 after CUDA_VISIBLE_DEVICES is set)
    device = get_device()
    
    # Create Nsight profiler if enabled
    nsight_profiler = None
    if profiling_config and profiling_config.get('enabled', False):
        nsight_profiler = NsightProfiler(
            output_dir=profiling_config.get('output_dir', 'results/profiles'),
            enabled=profiling_config.get('enabled', False),
            profile_iterations=profiling_config.get('profile_iterations', 10)
        )
    
    # Create Prometheus metrics exporter if enabled
    # Note: Each Ray task runs in its own process, so each will start its own HTTP server
    # This means multiple metrics endpoints (one per GPU/task). For production, consider
    # aggregating metrics or using a service mesh.
    prometheus_exporter = None
    if prometheus_config and prometheus_config.get('enabled', False):
        # Use a unique port per GPU to avoid conflicts (base port + gpu_id)
        base_port = prometheus_config.get('port', 8000)
        task_port = base_port + gpu_id
        prometheus_exporter = PrometheusMetricsExporter(
            port=task_port,
            enabled=prometheus_config.get('enabled', False)
        )
    
    # Create model wrapper
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
    
    # Create compiler
    if compiler_name == "pytorch_eager":
        compiler = PyTorchEagerCompiler()
    elif compiler_name == "torch_inductor":
        compiler = TorchInductorCompiler(mode="default")
    elif compiler_name == "torchscript" or compiler_name == "torchscript_trace":
        compiler = TorchScriptCompiler(method="trace")
    elif compiler_name == "torchscript_script":
        compiler = TorchScriptCompiler(method="script")
    elif compiler_name == "onnx_runtime":
        compiler = ONNXRuntimeCompiler()
    elif compiler_name == "tvm":
        compiler = TVMCompiler()
    elif compiler_name == "tvm_autotuned":
        compiler = TVMCompiler(use_autotuning=True)
    elif compiler_name == "tensorrt" or compiler_name == "tensorrt_fp32":
        compiler = TensorRTCompiler(fp16_mode=False)
    elif compiler_name == "tensorrt_fp16":
        compiler = TensorRTCompiler(fp16_mode=True)
    else:
        raise ValueError(f"Unknown compiler: {compiler_name}")
    
    # Run benchmark
    runner = BenchmarkRunner(
        device=device,
        warmup_iters=warmup_iters,
        measured_iters=measured_iters,
        nsight_profiler=nsight_profiler,
        prometheus_exporter=prometheus_exporter
    )
    
    return runner.run_benchmark(model_wrapper, compiler, batch_size)


class RayBenchmarkRunner:
    """Distributed benchmark runner using Ray for parallel execution across GPUs.
    
    Designed for single-node multi-GPU setup where each GPU runs a different
    benchmark task concurrently. Can also connect to existing Ray clusters
    (e.g., Kubernetes) for multi-node support.
    """
    
    def __init__(
        self,
        num_gpus: Optional[int] = None,
        num_cpus: Optional[int] = None,
        head_address: Optional[str] = None,
        resources_per_task: dict = None
    ):
        """Initialize Ray cluster and configure resources.
        
        Args:
            num_gpus: Number of GPUs to use (auto-detect if None)
            num_cpus: Number of CPUs to use (auto-detect if None)
            head_address: Address of existing Ray cluster (if connecting to existing)
            resources_per_task: Resources required per task (default: {'num_gpus': 1, 'num_cpus': 2})
        """
        self.resources_per_task = resources_per_task or {'num_gpus': 1, 'num_cpus': 2}
        
        # Check if running in Kubernetes
        is_k8s = os.environ.get('KUBERNETES_SERVICE_HOST') is not None
        
        if head_address:
            # Connect to existing cluster (e.g., Kubernetes)
            print(f"Connecting to existing Ray cluster at {head_address}")
            ray.init(address=head_address, ignore_reinit_error=True)
        elif is_k8s:
            # Running in Kubernetes - Ray should auto-discover
            print("Detected Kubernetes environment, connecting to Ray cluster")
            ray.init(address="auto", ignore_reinit_error=True)
        else:
            # Start local cluster (single-node multi-GPU)
            import torch
            if num_gpus is None:
                num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if num_cpus is None:
                import os
                num_cpus = os.cpu_count() or 1
            
            print(f"Initializing local Ray cluster with {num_gpus} GPUs, {num_cpus} CPUs")
            ray.init(
                num_gpus=num_gpus,
                num_cpus=num_cpus,
                ignore_reinit_error=True
            )
    
    def run_distributed_benchmarks(
        self,
        models_config: List[dict],
        compiler_names: List[str],
        warmup_iters: int,
        measured_iters: int,
        profiling_config: Optional[dict] = None,
        prometheus_config: Optional[dict] = None
    ) -> List[BenchmarkMetrics]:
        """Run benchmarks in parallel across available GPUs.
        
        Args:
            models_config: List of model configs, each with 'name' and either 'input_shape' or 'max_length'
            compiler_names: List of compiler names to test
            warmup_iters: Number of warmup iterations
            measured_iters: Number of measured iterations
            profiling_config: Optional dict with profiling settings (enabled, output_dir, profile_iterations)
        
        Returns:
            List of BenchmarkMetrics from all benchmark runs
        """
        # Generate all (model, compiler, batch_size) combinations
        tasks = []
        for model_cfg in models_config:
            model_name = model_cfg['name']
            batch_sizes = model_cfg['batch_sizes']
            
            # Prepare model config dict for serialization
            if 'input_shape' in model_cfg:
                model_config_dict = {'input_shape': model_cfg['input_shape']}
            elif 'max_length' in model_cfg:
                model_config_dict = {'max_length': model_cfg['max_length']}
            else:
                raise ValueError(f"Model {model_name} must have either input_shape or max_length")
            
            for compiler_name in compiler_names:
                for batch_size in batch_sizes:
                    tasks.append({
                        'model_name': model_name,
                        'compiler_name': compiler_name,
                        'batch_size': batch_size,
                        'model_config': model_config_dict
                    })
        
        # Assign GPUs to tasks
        num_gpus = ray.cluster_resources().get('GPU', 0)
        if num_gpus == 0:
            # Fallback to CPU if no GPUs detected
            gpu_assignments = [0] * len(tasks)
            print("Warning: No GPUs detected in Ray cluster, falling back to CPU")
        else:
            gpu_assignments = get_gpu_assignments(len(tasks), int(num_gpus))
        
        print(f"\n{'='*70}")
        print(f"Running {len(tasks)} benchmarks across {num_gpus} GPUs")
        print(f"{'='*70}")
        
        # Submit all tasks to Ray
        futures = []
        for i, task in enumerate(tasks):
            gpu_id = gpu_assignments[i]
            future = run_benchmark_task.remote(
                model_name=task['model_name'],
                compiler_name=task['compiler_name'],
                batch_size=task['batch_size'],
                model_config=task['model_config'],
                warmup_iters=warmup_iters,
                measured_iters=measured_iters,
                gpu_id=gpu_id,
                profiling_config=profiling_config,
                prometheus_config=prometheus_config
            )
            futures.append((future, task))
        
        # Collect results as they complete
        results = []
        for i, (future, task) in enumerate(futures):
            model_name = task['model_name']
            compiler_name = task['compiler_name']
            batch_size = task['batch_size']
            print(f"[{i+1}/{len(tasks)}] Waiting for: {model_name} | {compiler_name} | batch_size={batch_size}...")
            try:
                result = ray.get(future)
                results.append(result)
            except Exception as e:
                print(f"Error running benchmark for {model_name} | {compiler_name} | batch_size={batch_size}: {e}")
                # Continue with other tasks
        
        return results
    
    def shutdown(self):
        """Shutdown Ray cluster."""
        ray.shutdown()

