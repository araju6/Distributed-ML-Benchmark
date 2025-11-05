import os
import tempfile
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from .base import Compiler

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


class TensorRTModelWrapper(nn.Module):
    """Wrapper that makes TensorRT engine callable like a PyTorch model."""
    
    def __init__(self, engine, context, input_shape: Tuple, output_shape: Tuple):
        super().__init__()
        self.engine = engine
        self.context = context
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def forward(self, input_tensor: torch.Tensor):
        """Forward pass using TensorRT engine."""

        if isinstance(input_tensor, torch.Tensor):
            input_np = input_tensor.detach().cpu().numpy()
        else:
            input_np = np.array(input_tensor)
        

        np.copyto(self.inputs[0]['host'], input_np.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        output_np = self.outputs[0]['host'].reshape(self.output_shape)
        return torch.from_numpy(output_np).to(input_tensor.device if isinstance(input_tensor, torch.Tensor) else 'cpu')


class TensorRTCompiler(Compiler):
    
    def __init__(
        self,
        max_batch_size: int = 32,
        fp16_mode: bool = False,
        int8_mode: bool = False,
        workspace_size: int = 1 << 30,  # 1GB
        min_timing_iterations: int = 1,
        avg_timing_iterations: int = 8
    ):
        if not TENSORRT_AVAILABLE:
            raise ImportError(
                "TensorRT is not installed. "
                "Install TensorRT and pycuda. "
                "Note: TensorRT may have compatibility issues on P100 GPUs."
            )
        
        self.max_batch_size = max_batch_size
        self.fp16_mode = fp16_mode
        self.int8_mode = int8_mode
        self.workspace_size = workspace_size
        self.min_timing_iterations = min_timing_iterations
        self.avg_timing_iterations = avg_timing_iterations
        self.temp_dir = None
        self.logger = trt.Logger(trt.Logger.WARNING)
    
    def compile(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        model.eval()
        
        input_shape = tuple(example_input.shape)
        batch_size = input_shape[0]
        
        self.temp_dir = tempfile.mkdtemp()
        onnx_path = os.path.join(self.temp_dir, "model.onnx")
        
        try:
            torch.onnx.export(
                model,
                example_input,
                onnx_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                } if batch_size == 1 else None,  # Dynamic shapes for batch_size=1
                opset_version=11,  # TensorRT 7.x/8.x compatible
                do_constant_folding=True,
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to export model to ONNX: {e}")
        
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                errors = []
                for error in range(parser.num_errors):
                    errors.append(parser.get_error(error))
                raise RuntimeError(f"Failed to parse ONNX model: {errors}")
        
        config = builder.create_builder_config()
        config.max_workspace_size = self.workspace_size
        
        if self.fp16_mode:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("TensorRT: FP16 mode enabled")
            else:
                print("Warning: FP16 not supported on this platform, using FP32")
        
        if self.int8_mode:
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                print("Warning: INT8 mode requires calibration data (not implemented)")
            else:
                print("Warning: INT8 not supported on this platform")
        
        print("Building TensorRT engine... (this may take a few minutes)")
        try:
            if batch_size == 1:
                profile = builder.create_optimization_profile()
                profile.set_shape('input', (1, *input_shape[1:]), (self.max_batch_size, *input_shape[1:]), (self.max_batch_size, *input_shape[1:]))
                config.add_optimization_profile(profile)
            
            engine = builder.build_engine(network, config)
            
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
        except Exception as e:
            raise RuntimeError(f"TensorRT engine build failed: {e}. "
                             f"Note: Some ops may not be supported on P100 (compute capability 6.0).")
        
        context = engine.create_execution_context()
        
        output_shape = tuple(engine.get_binding_shape(1))  # Assuming single output
        if batch_size > 1:
            output_shape = (batch_size, *output_shape[1:])
        
        print(f"TensorRT engine built successfully")
        print(f"  Input shape: {input_shape}")
        print(f"  Output shape: {output_shape}")
        
        return TensorRTModelWrapper(engine, context, input_shape, output_shape)
    
    def get_name(self) -> str:
        precision = "fp32"
        if self.fp16_mode:
            precision = "fp16"
        elif self.int8_mode:
            precision = "int8"
        return f"tensorrt_{precision}"
    
    def supports_dynamic_shapes(self) -> bool:
        return True  # TensorRT supports dynamic batch sizes
    
    def __del__(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except:
                pass

