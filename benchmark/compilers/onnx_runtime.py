import os
import tempfile
import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from typing import Union
from .base import Compiler


class ONNXModelWrapper(nn.Module):
    """Wrapper that makes ONNX Runtime model callable like a PyTorch model."""
    def __init__(self, session: ort.InferenceSession, input_names: list, output_names: list):
        super().__init__()
        self.session = session
        self.input_names = input_names
        self.output_names = output_names
    
    def forward(self, *args, **kwargs):
        if args:
            inputs = args[0] if len(args) == 1 else args
        else:
            inputs = kwargs.get(self.input_names[0]) if self.input_names else None
        
        if inputs is None:
            raise ValueError("No input provided to ONNX model")
        
        if isinstance(inputs, torch.Tensor):
            inputs_np = inputs.detach().cpu().numpy()
        else:
            inputs_np = np.array(inputs)
        
        input_dict = {self.input_names[0]: inputs_np}
        
        outputs = self.session.run(self.output_names, input_dict)
        
        if len(outputs) == 1:
            return torch.from_numpy(outputs[0]).to(inputs.device if isinstance(inputs, torch.Tensor) else 'cpu')
        else:
            return tuple(torch.from_numpy(out).to(inputs.device if isinstance(inputs, torch.Tensor) else 'cpu') for out in outputs)


class ONNXRuntimeCompiler(Compiler):
    def __init__(self, provider: str = 'CUDAExecutionProvider', optimize: bool = True):
        self.provider = provider
        self.optimize = optimize
        self.temp_dir = None
        self.onnx_model_path = None
    
    def compile(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:

        model.eval()
        
        self.temp_dir = tempfile.mkdtemp()
        self.onnx_model_path = os.path.join(self.temp_dir, "model.onnx")
        
        try:
            torch.onnx.export(
                model,
                example_input,
                self.onnx_model_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                } if self.supports_dynamic_shapes() else None,
                opset_version=13,  # Good compatibility
                do_constant_folding=True,
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to export model to ONNX: {e}")
        
        sess_options = ort.SessionOptions()
        if self.optimize:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        try:
            providers = []
            if self.provider == 'CUDAExecutionProvider':
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.append('CUDAExecutionProvider')
                else:
                    print("Warning: CUDAExecutionProvider not available, falling back to CPU")
                    providers.append('CPUExecutionProvider')
            else:
                providers.append(self.provider)
            
            session = ort.InferenceSession(
                self.onnx_model_path,
                sess_options=sess_options,
                providers=providers
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create ONNX Runtime session: {e}")
        
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        
        return ONNXModelWrapper(session, input_names, output_names)
    
    def get_name(self) -> str:
        return "onnx_runtime"
    
    def supports_dynamic_shapes(self) -> bool:
        return True
    
    def __del__(self):
        if self.onnx_model_path and os.path.exists(self.onnx_model_path):
            try:
                os.remove(self.onnx_model_path)
                if self.temp_dir and os.path.exists(self.temp_dir):
                    os.rmdir(self.temp_dir)
            except:
                pass

