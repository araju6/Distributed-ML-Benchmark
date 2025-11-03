import torch
import torch.nn as nn
from .base import Compiler

class TorchInductorCompiler(Compiler):
    def __init__(self, mode="default"):
        self.mode = mode
        # Check if device supports Triton (CUDA capability >= 7.0)
        self._supports_triton = self._check_triton_support()
    
    def _check_triton_support(self) -> bool:
        """Check if the current device supports Triton compiler"""
        if not torch.cuda.is_available():
            return False
        try:
            device_props = torch.cuda.get_device_properties(0)
            compute_capability = device_props.major * 10 + device_props.minor
            return compute_capability >= 70  # 7.0
        except:
            return False
    
    def compile(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        if not self._supports_triton:
            # Fall back to eager mode for devices that don't support Triton
            # (e.g., P100 with compute capability 6.0)
            import warnings
            warnings.warn(
                f"Device does not support Triton compiler (CUDA capability < 7.0). "
                f"Falling back to eager mode for torch.compile.",
                UserWarning
            )
            # Return model as-is (eager mode)
            return model
        
        compiled_model = torch.compile(model, mode=self.mode)
        return compiled_model
    
    def get_name(self) -> str:
        if not self._supports_triton:
            return f"torch_inductor_{self.mode}_fallback_eager"
        return f"torch_inductor_{self.mode}"
    
    def supports_dynamic_shapes(self) -> bool:
        return True