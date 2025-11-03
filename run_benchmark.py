import torch
from benchmark.core.config import Config
from benchmark.core.benchmark_runner import BenchmarkRunner
from benchmark.models.resnet import ResNetWrapper
from benchmark.models.mobilenet import MobileNetWrapper
from benchmark.compilers.pytorch_eager import PyTorchEagerCompiler
from benchmark.compilers.torch_inductor import TorchInductorCompiler
from benchmark.compilers.torchscript import TorchScriptCompiler
from benchmark.utils.device import get_device
from benchmark.utils.output import ResultsWriter

def get_compiler(compiler_name: str):
    """Factory function to create compiler instances"""
    if compiler_name == "pytorch_eager":
        return PyTorchEagerCompiler()
    elif compiler_name == "torch_inductor":
        return TorchInductorCompiler(mode="default")
    elif compiler_name == "torchscript" or compiler_name == "torchscript_trace":
        return TorchScriptCompiler(method="trace")
    elif compiler_name == "torchscript_script":
        return TorchScriptCompiler(method="script")
    else:
        raise ValueError(f"Unknown compiler: {compiler_name}. Available: pytorch_eager, torch_inductor, torchscript")

def get_model(model_name: str, input_shape):
    """Factory function to create model instances"""
    if model_name == "resnet50":
        return ResNetWrapper(input_shape=tuple(input_shape), pretrained=True)
    elif model_name == "mobilenet_v3":
        return MobileNetWrapper(input_shape=tuple(input_shape), pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: resnet50, mobilenet_v3")

def main():
    config = Config.from_yaml("config.yaml")
    
    print("="*70)
    print("ML COMPILER BENCHMARK FRAMEWORK")
    print("="*70)
    print(f"Model: {config.model.name}")
    print(f"Compilers: {', '.join(config.compilers)}")
    print(f"Batch sizes: {config.model.batch_sizes}")
    print(f"Warmup iterations: {config.benchmark.warmup_iterations}")
    print(f"Measured iterations: {config.benchmark.measured_iterations}")
    print("="*70)
    
    device = get_device()
    
    runner = BenchmarkRunner(
        device=device,
        warmup_iters=config.benchmark.warmup_iterations,
        measured_iters=config.benchmark.measured_iterations
    )
    
    model_wrapper = get_model(config.model.name, config.model.input_shape)
    
    all_results = []
    
    for compiler_name in config.compilers:
        compiler = get_compiler(compiler_name)
        
        for batch_size in config.model.batch_sizes:
            metrics = runner.run_benchmark(model_wrapper, compiler, batch_size)
            all_results.append(metrics)
    
    output_path = f"{config.output.save_path}/benchmark_results.csv"
    ResultsWriter.write_csv(all_results, output_path)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
