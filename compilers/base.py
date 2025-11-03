from abc import ABC, abstractmethod
import torch.nn as nn

class Compiler(ABC):
    """abstract base class for all theML compilers"""
    
    @abstractmethod
    def compile(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    def supports_dynamic_shapes(self) -> bool:
        return False