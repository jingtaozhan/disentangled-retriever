import torch
from torch import Tensor
import torch.nn.functional as F

from transformers import AutoConfig

from .bert_dense import BertDense
from .roberta_dense import RobertaDense
from .distilbert_dense import DistilBertDense
from .utils import SIMILARITY_METRICS, POOLING_METHODS

class AutoDenseModel:
    @classmethod
    def from_pretrained(cls, model_path: str, config = None):
        if config is None:
            config = AutoConfig.from_pretrained(model_path)
        if config.model_type == "bert":
            model = BertDense.from_pretrained(model_path, config=config)
        elif config.model_type == "roberta":
            model = RobertaDense.from_pretrained(model_path, config=config)
        elif config.model_type == "distilbert":
            model = DistilBertDense.from_pretrained(model_path, config=config)
        else:
            raise NotImplementedError()
        return model
