import torch
from benchmark.core.config import Config
from benchmark.core.benchmark_runner import BenchmarkRunner
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
    directory. Nothing fancy, just orchestration.
    """
    cfg = Config.from_yaml("config.yaml")
    
    print("="*70)
    print("ML COMPILER BENCHMARK FRAMEWORK")
    print("="*70)
    print(f"Models: {', '.join([m.name for m in cfg.models])}")
    print(f"Compilers: {', '.join(cfg.compilers)}")
    print(f"Warmup iterations: {cfg.benchmark.warmup_iterations}")
    print(f"Measured iterations: {cfg.benchmark.measured_iterations}")
    print("="*70)
    
    device = get_device()
    
    runner = BenchmarkRunner(
        device=device,
        warmup_iters=cfg.benchmark.warmup_iterations,
        measured_iters=cfg.benchmark.measured_iterations
    )
    
    combined_results = []
    
    # Iterate over all models
    for model_config in cfg.models:
        print(f"\n{'#'*70}")
        print(f"Processing model: {model_config.name}")
        print(f"{'#'*70}")
        
        model_wrapper = get_model(model_config)
        
        # Iterate over compilers
        for compiler_name in cfg.compilers:
            compiler = get_compiler(compiler_name)
            
            # Iterate over batch sizes for this model
            for batch_size in model_config.batch_sizes:
                run_stats = runner.run_benchmark(model_wrapper, compiler, batch_size)
                combined_results.append(run_stats)
    
    output_path = f"{cfg.output.save_path}/benchmark_results.csv"
    ResultsWriter.write_csv(combined_results, output_path)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print(f"Results saved to: {output_path}")
    print("="*70)

if __name__ == "__main__":
    main()
