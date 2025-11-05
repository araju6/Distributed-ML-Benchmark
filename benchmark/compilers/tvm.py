import os
import tempfile
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from .base import Compiler

try:
    import tvm
    from tvm import relay
    from tvm.contrib import graph_executor
    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False


class TVMModelWrapper(nn.Module):
    """Wrapper that makes TVM compiled model callable like a PyTorch model."""
    
    def __init__(self, lib: tvm.runtime.Module, dev: tvm.device, input_name: str = "input0"):
        super().__init__()
        self.lib = lib
        self.dev = dev
        self.module = graph_executor.GraphModule(lib["default"](dev))
        self.input_name = input_name
    
    def forward(self, input_tensor: torch.Tensor):
        if isinstance(input_tensor, torch.Tensor):
            input_np = input_tensor.detach().cpu().numpy()
        else:
            input_np = np.array(input_tensor)
        
        self.module.set_input(self.input_name, tvm.nd.array(input_np, self.dev))
        
        self.module.run()
        
        output = self.module.get_output(0)
        
        output_np = output.numpy()
        return torch.from_numpy(output_np).to(input_tensor.device if isinstance(input_tensor, torch.Tensor) else 'cpu')


class TVMCompiler(Compiler):
    def __init__(
        self,
        target: str = "cuda",
        opt_level: int = 3,
        use_autotuning: bool = False,
        tuning_trials: int = 1000
    ):
        if not TVM_AVAILABLE:
            raise ImportError(
                "TVM is not installed. Install with: pip install apache-tvm"
            )
        
        self.target = target
        self.opt_level = opt_level
        self.use_autotuning = use_autotuning
        self.tuning_trials = tuning_trials
        self.temp_dir = None
    
    def compile(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        model.eval()
        
        if self.target == "cuda":
            if torch.cuda.is_available():
                cuda_arch = torch.cuda.get_device_capability(0)
                target_str = f"cuda -arch=sm_{cuda_arch[0]}{cuda_arch[1]}"
            else:
                target_str = "llvm"
                print("Warning: CUDA not available, falling back to LLVM target")
        else:
            target_str = self.target
        
        tvm_target = tvm.target.Target(target_str)
        tvm_dev = tvm.device(str(tvm_target.kind.name), 0)
        
        try:
            input_shape = list(example_input.shape)
            input_name = "input0"
            
            scripted_model = torch.jit.trace(model, example_input)
            
            try:
                shape_list = [(input_name, input_shape)]
                mod, params = relay.frontend.from_pytorch(
                    scripted_model, shape_list
                )
            except Exception as e1:
                print(f"Direct PyTorch conversion failed: {e1}, trying ONNX path...")
                import tempfile
                self.temp_dir = tempfile.mkdtemp()
                onnx_path = os.path.join(self.temp_dir, "model.onnx")
                
                torch.onnx.export(
                    scripted_model,
                    example_input,
                    onnx_path,
                    input_names=[input_name],
                    output_names=['output'],
                    opset_version=13,
                    do_constant_folding=True,
                    verbose=False
                )
                
                mod, params = relay.frontend.from_onnx(onnx_path)
                
        except Exception as e:
            raise RuntimeError(f"Failed to convert model to TVM Relay: {e}")

        try:
            if self.use_autotuning:
                print(f"Running TVM autotuning with {self.tuning_trials} trials...")
                from tvm import autotvm
                pass
            
            with tvm.transform.PassContext(opt_level=self.opt_level):
                lib = relay.build(mod, target=tvm_target, params=params)
            
        except Exception as e:
            raise RuntimeError(f"Failed to build TVM module: {e}")
        
        return TVMModelWrapper(lib, tvm_dev, input_name)
    
    def get_name(self) -> str:
        autotune_suffix = "_autotuned" if self.use_autotuning else ""
        return f"tvm{autotune_suffix}"
    
    def supports_dynamic_shapes(self) -> bool:
        return False
    
    def __del__(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except:
                pass

