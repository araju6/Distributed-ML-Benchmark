import torch
import torch.nn as nn
from .base import Compiler

class TorchInductorCompiler(Compiler):
    def __init__(self, mode="default"):
        self.mode = mode
    
    def compile(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        compiled_model = torch.compile(model, mode=self.mode)
        return compiled_model
    
    def get_name(self) -> str:
        return f"torch_inductor_{self.mode}"
    
    def supports_dynamic_shapes(self) -> bool:
        return True