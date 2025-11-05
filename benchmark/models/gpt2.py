import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .base import ModelWrapper
import random


class GPT2ModelWrapper(nn.Module):
    
    def __init__(self, gpt2_model):
        super().__init__()
        self.gpt2_model = gpt2_model
    
    def forward(self, input_ids):
        outputs = self.gpt2_model(input_ids)
        return outputs.logits


class GPT2Wrapper(ModelWrapper):

    def __init__(self, max_length=128, model_size="gpt2", pretrained=True):
        self.max_length = max_length
        self.model_size = model_size
        
        if pretrained:
            gpt2_model = GPT2LMHeadModel.from_pretrained(self.model_size)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_size)
        else:
            from transformers import GPT2Config
            config = GPT2Config.from_pretrained(self.model_size)
            gpt2_model = GPT2LMHeadModel(config)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_size)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2ModelWrapper(gpt2_model)
        self.model.eval()
    
    def get_model(self) -> nn.Module:
        return self.model
    
    def get_example_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "GPT-2 is a transformer model developed by OpenAI.",
            "Artificial intelligence is transforming the world.",
            "PyTorch and TensorFlow are popular deep learning frameworks.",
            "This is an example sentence to test GPT-2 model."
        ]
        
        text = random.choice(sample_texts)
        
        encoded = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        
        if batch_size > 1:
            input_ids = input_ids.repeat(batch_size, 1)
        
        return input_ids
    
    def get_name(self) -> str:
        return f"{self.model_size}_seq{self.max_length}"

