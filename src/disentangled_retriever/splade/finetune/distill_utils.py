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


class BaseDistillSpladeFinetuner:
    
    # @staticmethod
    # def _flops(inputs):
    #     return torch.sum(torch.mean(torch.abs(inputs), dim=0) ** 2)

    def get_quadratic_increase_flop_factor(self, flop_factor):
        if self.state.epoch  >= self.args.flop_increase_epoch_factor:
            return flop_factor
        else:
            return flop_factor * (self.state.epoch / self.args.flop_increase_epoch_factor) ** 2

    def maybe_log_flop(self, q_factor, p_factor, q_flops, p_flops, rank_loss, scaled_flops):
        if self.args.flop_log_steps > 0 and self.state.global_step  % self.args.flop_log_steps == 0:
            q_factor, p_factor, q_flops, p_flops = float(q_factor), float(p_factor), float(q_flops), float(p_flops)
            self.log({
                "q_factor": round(q_factor, 4),
                "p_factor": round(p_factor, 4),
                "q_flops": round(q_flops, 2),
                "d_flops": round(p_flops, 2),
                "rank_loss": round(float(rank_loss), 2),
                "scaled_flops": round(float(scaled_flops), 2),
            })

    @staticmethod
    def _flops(*inputs):
        sum_distrib = torch.sum(torch.abs(inputs[0]), dim=0)
        tensor_num = len(inputs[0])
        for tensor in inputs[1:]:
            sum_distrib = sum_distrib + torch.sum(torch.abs(tensor), dim=0)
            tensor_num += len(tensor)
        average_distrib = sum_distrib / tensor_num
        return torch.sum(average_distrib ** 2)

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
        doc_embeds = model(**inputs['doc_input'])

        q_flops_loss_factor = self.get_quadratic_increase_flop_factor(self.args.q_flops_loss_factor)
        p_flops_loss_factor = self.get_quadratic_increase_flop_factor(self.args.p_flops_loss_factor )

        if self.args.local_rank > -1:
            q_flops_loss = self._flops(
                *self._gather_tensor_without_cat(query_embeds))
            p_flops_loss = self._flops(
                *self._gather_tensor_without_cat(doc_embeds)) 
            flops_loss = (q_flops_loss_factor * q_flops_loss + p_flops_loss_factor * p_flops_loss) * dist.get_world_size()
        else:
            q_flops_loss = self._flops(query_embeds)
            p_flops_loss = self._flops(doc_embeds)
            flops_loss = q_flops_loss_factor * q_flops_loss + p_flops_loss_factor * p_flops_loss
        
        doc_embeds = doc_embeds.reshape(len(query_embeds), (1 + self.args.neg_per_query), -1) # Nq, (1+n), dim
        assert doc_embeds.size(1) == 1 + self.args.neg_per_query, f"{doc_embeds.shape}, {self.args.neg_per_query}"
        model_scores = torch.matmul(query_embeds[:, None, :], doc_embeds.transpose(-2, -1)).squeeze(1) * self.args.inv_temperature # Nq, 1+n

        margin_preds = _compute_margins(model_scores) # Nq, n
        margin_labels = _compute_margins(inputs['ce_scores']) # Nq, n
        rank_loss = nn.MSELoss()(margin_preds, margin_labels)

        loss = rank_loss + flops_loss
        self.maybe_log_flop(q_flops_loss_factor, p_flops_loss_factor, q_flops_loss, p_flops_loss, rank_loss, flops_loss)
        return (loss, (query_embeds, doc_embeds, model_scores)) if return_outputs else loss

    def _gather_tensor_without_cat(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.args.local_rank] = t
        # cat_tensors = torch.cat(all_tensors)
        return all_tensors

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute Margin MSE loss.
        """
        loss, (query_embeds, doc_embeds, model_scores) = self._compute_distill_loss(model, inputs, True)
        return (loss, (query_embeds, doc_embeds)) if return_outputs else loss
    
    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        return 0


class BackboneDistillSpladeFinetuner(BaseDistillSpladeFinetuner, Trainer):
    pass


class AdapterDistillSpladeFinetuner(BaseDistillSpladeFinetuner, AdapterTrainer):
    
    def _load_from_checkpoint(self, resume_from_checkpoint):
        super()._load_from_checkpoint(resume_from_checkpoint)
        self._move_model_to_device(self.model, self.args.device)
