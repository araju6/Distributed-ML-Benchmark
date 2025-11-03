import torch
import torch.nn as nn
from .base import Compiler

class TorchScriptCompiler(Compiler):
    """
    TorchScript compiler using torch.jit.trace
    Works on all CUDA devices including P100
    """
    def __init__(self, method="trace"):
        """
        Args:
            method: "trace" or "script"
            - trace: Records operations on example input (faster, less flexible)
            - script: Compiles Python code directly (slower, more flexible)
        """
        self.method = method
    
    def compile(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        model.eval()
        
        if self.method == "trace":
            # Trace the model with example input
            traced_model = torch.jit.trace(model, example_input)
            # Optimize for inference
            traced_model = torch.jit.optimize_for_inference(traced_model)
            return traced_model
        
        elif self.method == "script":
            # Script the model (more flexible but slower to compile)
            scripted_model = torch.jit.script(model)
            scripted_model = torch.jit.optimize_for_inference(scripted_model)
            return scripted_model
        
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'trace' or 'script'")
    
    def get_name(self) -> str:
        return f"torchscript_{self.method}"
    
    def supports_dynamic_shapes(self) -> bool:
        # TorchScript trace doesn't support dynamic shapes well
        # Script mode has better support but still limited
        return False

