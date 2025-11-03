import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from .base import ModelWrapper

class MobileNetWrapper(ModelWrapper):
    """MobileNetV3 Large model wrapper"""
    
    def __init__(self, input_shape=(3, 224, 224), pretrained=True):
        self.input_shape = input_shape
        if pretrained:
            self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        else:
            self.model = mobilenet_v3_large(weights=None)
        
        self.model.eval()
    
    def get_model(self) -> nn.Module:
        return self.model
    
    def get_example_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, *self.input_shape, device=device)
    
    def get_name(self) -> str:
        return "mobilenet_v3_large"

