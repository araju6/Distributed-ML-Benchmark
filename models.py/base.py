from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class ModelWrapper(ABC):
    """abstract base class for model wrappers"""
    
    @abstractmethod
    def get_model(self) -> nn.Module:
        pass
    
    @abstractmethod
    def get_example_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass