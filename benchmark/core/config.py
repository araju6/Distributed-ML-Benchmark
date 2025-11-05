from dataclasses import dataclass, field
from typing import List, Optional
import yaml

@dataclass
class BenchmarkConfig:
    warmup_iterations: int
    measured_iterations: int

@dataclass
class ModelConfig:
    name: str
    batch_sizes: List[int]
    precision: str
    input_shape: Optional[List[int]] = None  # For vision models (C, H, W)
    max_length: Optional[int] = None  # For NLP models (sequence length)
    
    def __post_init__(self):
        """Validate that either input_shape or max_length is provided."""
        if self.input_shape is None and self.max_length is None:
            raise ValueError("Either input_shape or max_length must be provided for model config")

@dataclass
class OutputConfig:
    format: str
    save_path: str

@dataclass
class Config:
    benchmark: BenchmarkConfig
    models: List[ModelConfig]  # Changed from single model to list
    compilers: List[str]
    output: OutputConfig
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Handle both old format (single model) and new format (models list)
        if 'model' in data:
            # Legacy: single model
            model_data = data['model']
            models_data = [model_data]
        else:
            # New: list of models
            models_data = data['models']
        
        models = [ModelConfig(**model_data) for model_data in models_data]
        
        return cls(
            benchmark=BenchmarkConfig(**data['benchmark']),
            models=models,
            compilers=data['compilers'],
            output=OutputConfig(**data['output'])
        )