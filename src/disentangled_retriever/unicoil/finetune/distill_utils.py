import torch
import random
import logging
import gzip, pickle
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Any, Union, Optional

from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset

from transformers import AdapterTrainer, PreTrainedTokenizer, Trainer

from ...dense.finetune.distill_utils import FinetuneCollator, QDRelDataset

logger = logging.getLogger(__name__)


class BaseDistillUnicoilFinetuner:
    def _compute_distill_loss(self, model, inputs, return_outputs=False):
        """
        Compute Margin MSE loss.
        """
        def _compute_margins(one_pos_multi_neg_scores):
            pos_scores = one_pos_multi_neg_scores[:, :1] # Nq, 1
            neg_scores = one_pos_multi_neg_scores[:, 1:] # Nq, n
            margin = pos_scores.expand_as(neg_scores) - neg_scores
            return margin # Nq, n
        
        query_embeds = model(**inputs['query_input']) # Nq, dim
        doc_embeds = model(**inputs['doc_input']).reshape(len(query_embeds), (1 + self.args.neg_per_query), -1) # Nq, (1+n), dim
        assert doc_embeds.size(1) == 1 + self.args.neg_per_query, f"{doc_embeds.shape}, {self.args.neg_per_query}"
        model_scores = torch.matmul(query_embeds[:, None, :], doc_embeds.transpose(-2, -1)).squeeze(1) * self.args.inv_temperature # Nq, 1+n

        margin_preds = _compute_margins(model_scores) # Nq, n
        margin_labels = _compute_margins(inputs['ce_scores']) # Nq, n
        loss = nn.MSELoss()(margin_preds, margin_labels)
        return (loss, (query_embeds, doc_embeds, model_scores)) if return_outputs else loss

    def _gather_tensor(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.args.local_rank] = t
        cat_tensors = torch.cat(all_tensors)
        return cat_tensors

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute Margin MSE loss.
        """
        loss, (query_embeds, doc_embeds, model_scores) = self._compute_distill_loss(model, inputs, True)
        return (loss, (query_embeds, doc_embeds)) if return_outputs else loss
    
    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        return 0


class BackboneDistillUnicoilFinetuner(BaseDistillUnicoilFinetuner, Trainer):
    pass


class AdapterDistillUnicoilFinetuner(BaseDistillUnicoilFinetuner, AdapterTrainer):
    
    def _load_from_checkpoint(self, resume_from_checkpoint):
        super()._load_from_checkpoint(resume_from_checkpoint)
        self._move_model_to_device(self.model, self.args.device)
