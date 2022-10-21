import torch
from torch import Tensor
import torch.nn.functional as F

from transformers import AutoConfig

from .bert_colbert import ColBERT

class AutoColBERTModel:
    @classmethod
    def from_pretrained(cls, model_path: str, config = None):
        if config is None:
            config = AutoConfig.from_pretrained(model_path)
        if config.model_type == "bert":
            model = ColBERT.from_pretrained(model_path, config=config)
        else:
            raise NotImplementedError()
        return model
