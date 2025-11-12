"""Factory functions for creating compilers and model wrappers.

This module provides centralized factory functions to avoid code duplication
across different entry points (run_benchmark.py, run_autocompiler.py, etc.).
"""

from typing import Union, Dict, Any
from ..compilers.base import Compiler
from ..models.base import ModelWrapper
from ..compilers.pytorch_eager import PyTorchEagerCompiler
from ..compilers.torch_inductor import TorchInductorCompiler
from ..compilers.torchscript import TorchScriptCompiler
from ..compilers.onnx_runtime import ONNXRuntimeCompiler
from ..models.resnet import ResNetWrapper
from ..models.mobilenet import MobileNetWrapper
from ..models.bert import BERTWrapper
from ..models.gpt2 import GPT2Wrapper


def get_compiler(compiler_name: str) -> Compiler:
    """Get compiler instance from name.
    
    Args:
        compiler_name: Name of the compiler (e.g., "pytorch_eager", "torchscript", etc.)
    
    Returns:
        Compiler instance
    
    Raises:
        ValueError: If compiler name is unknown or optional dependencies are missing
    """
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
        try:
            from ..compilers.tvm import TVMCompiler
            return TVMCompiler()
        except ImportError as e:
            raise ValueError(f"TVM not available: {e}")
    elif compiler_name == "tvm_autotuned":
        try:
            from ..compilers.tvm import TVMCompiler
            return TVMCompiler(use_autotuning=True)
        except ImportError as e:
            raise ValueError(f"TVM not available: {e}")
    elif compiler_name == "tensorrt" or compiler_name == "tensorrt_fp32":
        try:
            from ..compilers.tensorrt import TensorRTCompiler
            return TensorRTCompiler(fp16_mode=False)
        except ImportError as e:
            raise ValueError(f"TensorRT not available: {e}")
    elif compiler_name == "tensorrt_fp16":
        try:
            from ..compilers.tensorrt import TensorRTCompiler
            return TensorRTCompiler(fp16_mode=True)
        except ImportError as e:
            raise ValueError(f"TensorRT not available: {e}")
    else:
        raise ValueError(
            f"Unknown compiler: {compiler_name}. "
            f"Available: pytorch_eager, torch_inductor, torchscript, torchscript_script, "
            f"onnx_runtime, tvm, tvm_autotuned, tensorrt, tensorrt_fp16"
        )


def get_model(
    model_name_or_config: Union[str, Any],
    input_format: Union[Dict[str, Any], None] = None,
    model_config: Union[Any, None] = None
) -> ModelWrapper:
    """Get model wrapper from name and configuration.
    
    Supports multiple input formats:
    1. model_name (str) + input_format dict (from run_autocompiler.py, profile_benchmark.py)
    2. model_config object with name attribute (from run_benchmark.py)
    
    Args:
        model_name_or_config: Either model name (str) or ModelConfig object
        input_format: Optional dict with 'input_shape' (list) or 'max_length' (int)
        model_config: Optional ModelConfig object (deprecated, use model_name_or_config)
    
    Returns:
        ModelWrapper instance
    
    Raises:
        ValueError: If model name is unknown or required config is missing
    """
    # Handle different call patterns
    if isinstance(model_name_or_config, str):
        # Pattern 1: get_model(model_name, input_format=dict)
        model_name = model_name_or_config
        if input_format is not None:
            input_shape = input_format.get('input_shape')
            max_length = input_format.get('max_length')
        elif model_config is not None:
            # Fallback to model_config if provided
            input_shape = getattr(model_config, 'input_shape', None)
            max_length = getattr(model_config, 'max_length', None)
        else:
            input_shape = None
            max_length = None
    else:
        # Pattern 2: get_model(model_config_object) - from run_benchmark.py
        model_config_obj = model_name_or_config
        model_name = model_config_obj.name
        input_shape = getattr(model_config_obj, 'input_shape', None)
        max_length = getattr(model_config_obj, 'max_length', None)
    
    # Create model based on name
    if model_name == "resnet50":
        if input_shape is None:
            raise ValueError("resnet50 requires input_shape")
        return ResNetWrapper(input_shape=tuple(input_shape), pretrained=True)
    elif model_name == "mobilenet_v3_large":
        if input_shape is None:
            raise ValueError("mobilenet_v3_large requires input_shape")
        return MobileNetWrapper(input_shape=tuple(input_shape), pretrained=True)
    elif model_name == "bert_base":
        if max_length is None:
            raise ValueError("bert_base requires max_length")
        return BERTWrapper(max_length=max_length, pretrained=True)
    elif model_name == "gpt2":
        if max_length is None:
            raise ValueError("gpt2 requires max_length")
        return GPT2Wrapper(max_length=max_length, pretrained=True)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: resnet50, mobilenet_v3_large, bert_base, gpt2"
        )

