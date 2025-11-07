import argparse
import torch
from benchmark.core.config import Config
from benchmark.core.benchmark_runner import BenchmarkRunner
from benchmark.core.ray_runner import RayBenchmarkRunner
from benchmark.core.nsight_profiler import NsightProfiler
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
from benchmark.utils.device import get_device
from benchmark.utils.output import ResultsWriter

def get_compiler(compiler_name: str):
    if compiler_name == "pytorch_eager":
        return PyTorchEagerCompiler()
    elif compiler_name == "torch_inductor":
        return TorchInductorCompiler(mode="default")
    elif compiler_name == "torchscript" or compiler_name == "torchscript_trace":
        return TorchScriptCompiler(method="trace")
    elif compiler_name == "torchscript_script":
        return TorchScriptCompiler(method="script")
    elif compiler_name == "onnx_runtime":
        return ONNXRuntimeCompiler()
    elif compiler_name == "tvm":
        return TVMCompiler()
    elif compiler_name == "tvm_autotuned":
        return TVMCompiler(use_autotuning=True)
    elif compiler_name == "tensorrt" or compiler_name == "tensorrt_fp32":
        return TensorRTCompiler(fp16_mode=False)
    elif compiler_name == "tensorrt_fp16":
        return TensorRTCompiler(fp16_mode=True)
    else:
        raise ValueError(
            f"Unknown compiler: {compiler_name}. "
            f"Available: pytorch_eager, torch_inductor, torchscript, onnx_runtime, tvm, tvm_autotuned, tensorrt, tensorrt_fp16"
        )

def get_model(model_config):
    """Create model wrapper from config.
    
    Args:
        model_config: ModelConfig object with name, input_shape/max_length, etc.
    
    Returns:
        ModelWrapper instance
    """
    model_name = model_config.name
    
    if model_name == "resnet50":
        if model_config.input_shape is None:
            raise ValueError("resnet50 requires input_shape")
        return ResNetWrapper(input_shape=tuple(model_config.input_shape), pretrained=True)
    elif model_name == "mobilenet_v3_large":
        if model_config.input_shape is None:
            raise ValueError("mobilenet_v3_large requires input_shape")
        return MobileNetWrapper(input_shape=tuple(model_config.input_shape), pretrained=True)
    elif model_name == "bert_base":
        if model_config.max_length is None:
            raise ValueError("bert_base requires max_length")
        return BERTWrapper(max_length=model_config.max_length, pretrained=True)
    elif model_name == "gpt2":
        if model_config.max_length is None:
            raise ValueError("gpt2 requires max_length")
        return GPT2Wrapper(max_length=model_config.max_length, pretrained=True)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: resnet50, mobilenet_v3_large, bert_base, gpt2"
        )

def main():
    """Entry point that loads config, runs all requested cases, and saves CSV.

    Reads `config.yaml`, builds model wrappers, iterates over models, compilers and
    batch sizes, and writes a single results file under the configured output
    directory. Supports both sequential and distributed (Ray) execution modes.
    """
    parser = argparse.ArgumentParser(description='ML Compiler Benchmark Framework')
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Enable distributed execution with Ray (multi-GPU parallel)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    args = parser.parse_args()
    
    cfg = Config.from_yaml(args.config)
    
    use_ray = args.distributed or cfg.ray.enabled
    
    print("="*70)
    print("ML COMPILER BENCHMARK FRAMEWORK")
    print("="*70)
    print(f"Execution mode: {'Distributed (Ray)' if use_ray else 'Sequential'}")
    print(f"Models: {', '.join([m.name for m in cfg.models])}")
    print(f"Compilers: {', '.join(cfg.compilers)}")
    print(f"Warmup iterations: {cfg.benchmark.warmup_iterations}")
    print(f"Measured iterations: {cfg.benchmark.measured_iterations}")
    print("="*70)
    
    if use_ray:
        try:
            ray_runner = RayBenchmarkRunner(
                num_gpus=cfg.ray.num_gpus,
                num_cpus=cfg.ray.num_cpus,
                head_address=cfg.ray.head_address,
                resources_per_task=cfg.ray.resources_per_task
            )
            
            models_config = []
            for model_cfg in cfg.models:
                model_dict = {
                    'name': model_cfg.name,
                    'batch_sizes': model_cfg.batch_sizes
                }
                if model_cfg.input_shape:
                    model_dict['input_shape'] = model_cfg.input_shape
                if model_cfg.max_length:
                    model_dict['max_length'] = model_cfg.max_length
                models_config.append(model_dict)
            
            # Prepare profiling config for Ray tasks
            profiling_config = None
            if cfg.profiling.enabled:
                profiling_config = {
                    'enabled': cfg.profiling.enabled,
                    'output_dir': cfg.profiling.output_dir,
                    'profile_iterations': cfg.profiling.profile_iterations
                }
            
            combined_results = ray_runner.run_distributed_benchmarks(
                models_config=models_config,
                compiler_names=cfg.compilers,
                warmup_iters=cfg.benchmark.warmup_iterations,
                measured_iters=cfg.benchmark.measured_iterations,
                profiling_config=profiling_config
            )
            
            ray_runner.shutdown()
            
        except ImportError:
            print("Error: Ray is not installed. Install with: pip install ray")
            print("Falling back to sequential execution...")
            use_ray = False
        except Exception as e:
            print(f"Error initializing Ray: {e}")
            print("Falling back to sequential execution...")
            use_ray = False
    
    if not use_ray:
        device = get_device()
        
        # Create Nsight profiler if enabled
        nsight_profiler = None
        if cfg.profiling.enabled:
            nsight_profiler = NsightProfiler(
                output_dir=cfg.profiling.output_dir,
                enabled=cfg.profiling.enabled,
                profile_iterations=cfg.profiling.profile_iterations
            )
        
        runner = BenchmarkRunner(
            device=device,
            warmup_iters=cfg.benchmark.warmup_iterations,
            measured_iters=cfg.benchmark.measured_iterations,
            nsight_profiler=nsight_profiler
        )
        
        combined_results = []
        
        for model_config in cfg.models:
            print(f"\n{'#'*70}")
            print(f"Processing model: {model_config.name}")
            print(f"{'#'*70}")
            
            model_wrapper = get_model(model_config)
            
            for compiler_name in cfg.compilers:
                compiler = get_compiler(compiler_name)
                
                for batch_size in model_config.batch_sizes:
                    run_stats = runner.run_benchmark(model_wrapper, compiler, batch_size)
                    combined_results.append(run_stats)
    
    output_path = f"{cfg.output.save_path}/benchmark_results.csv"
    ResultsWriter.write_csv(combined_results, output_path)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print(f"Results saved to: {output_path}")
    print(f"Total benchmarks completed: {len(combined_results)}")
    print("="*70)

if __name__ == "__main__":
    main()
