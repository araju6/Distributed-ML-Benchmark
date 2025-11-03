import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from .base import ModelWrapper

class ResNetWrapper(ModelWrapper):
    
    def __init__(self, input_shape=(3, 224, 224), pretrained=True):
        self.input_shape = input_shape
        if pretrained:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.model = resnet50(weights=None)
        
        self.model.eval()
    
    def get_model(self) -> nn.Module:
        return self.model
    
    def get_example_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, *self.input_shape, device=device)
    
    def get_name(self) -> str:
        return "resnet50"