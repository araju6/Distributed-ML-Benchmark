import torch
import torch.nn as nn
from .base import Compiler

class PyTorchEagerCompiler(Compiler):
    
    def compile(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        return model
    
    def get_name(self) -> str:
        return "pytorch_eager"
    
    def supports_dynamic_shapes(self) -> bool:
        return True