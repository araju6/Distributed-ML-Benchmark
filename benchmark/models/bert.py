import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from .base import ModelWrapper
import random

class BERTModelWrapper(nn.Module):
    
    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = bert_model
    
    def forward(self, input_ids):
        outputs = self.bert_model(input_ids)
        return outputs.pooler_output


class BERTWrapper(ModelWrapper):
    
    def __init__(self, max_length=128, pretrained=True):
        self.max_length = max_length
        self.model_name = "bert-base-uncased"
        
        if pretrained:
            bert_model = BertModel.from_pretrained(self.model_name)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        else:
            from transformers import BertConfig
            config = BertConfig.from_pretrained(self.model_name)
            bert_model = BertModel(config)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        self.model = BERTModelWrapper(bert_model)
        self.model.eval()
    
    def get_model(self) -> nn.Module:
        return self.model
    
    def get_example_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "BERT is a transformer model developed by Google.",
            "Artificial intelligence is transforming the world.",
            "PyTorch and TensorFlow are popular deep learning frameworks.",
            "This is an example sentence to test BERT model."
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
        return f"bert_base_seq{self.max_length}"

