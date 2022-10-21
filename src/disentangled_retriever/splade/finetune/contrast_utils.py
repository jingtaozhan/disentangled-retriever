import os
import torch
import random
import logging
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional

from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset

from transformers import AdapterTrainer, PreTrainedTokenizer, Trainer

from ...dense.finetune.contrast_utils import FinetuneCollator, QDRelDataset

logger = logging.getLogger(__name__)


class BaseContrastSpladeFinetuner:

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

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute contrastive loss.
        """
        query_embeds = model(**inputs['query_input']) # Nq, dim
        doc_embeds = model(**inputs['doc_input']) # Nq, dim
        qids = self._prepare_input(inputs['qids']).contiguous() # _prepare_input to gpu
        docids = self._prepare_input(inputs['docids']).contiguous()

        q_flops_loss_factor = self.get_quadratic_increase_flop_factor(self.args.q_flops_loss_factor)
        p_flops_loss_factor = self.get_quadratic_increase_flop_factor(self.args.p_flops_loss_factor )
        if self.args.local_rank > -1:
            query_embeds = self._gather_tensor(query_embeds)
            doc_embeds = self._gather_tensor(doc_embeds)
            qids, docids = self._gather_tensor(qids), self._gather_tensor(docids) 

        q_flops_loss = self._flops(query_embeds)
        
        if 'neg_doc_input' not in inputs:
            rank_loss = self.compute_inbatch_contrastive_loss(query_embeds, doc_embeds, qids, docids)
            p_flops_loss = self._flops(doc_embeds)
            flops_loss = q_flops_loss_factor * q_flops_loss + p_flops_loss_factor * p_flops_loss
            if self.args.local_rank > -1:
                flops_loss = flops_loss * dist.get_world_size()
            self.maybe_log_flop(q_flops_loss_factor, p_flops_loss_factor, q_flops_loss, p_flops_loss, rank_loss, flops_loss)
            loss = rank_loss + flops_loss
            return (loss, (query_embeds, doc_embeds)) if return_outputs else loss
        else:
            neg_doc_embeds = model(**inputs['neg_doc_input']) 
            neg_docids = self._prepare_input(inputs['neg_docids']).contiguous()
            if self.args.local_rank > -1:
                neg_doc_embeds = self._gather_tensor(neg_doc_embeds)
                neg_docids = self._gather_tensor(neg_docids)
            rank_loss = self.compute_contrastive_loss(query_embeds, doc_embeds, neg_doc_embeds, qids, docids, neg_docids)
            p_flops_loss = self._flops(doc_embeds, neg_doc_embeds)
            flops_loss = q_flops_loss_factor * q_flops_loss + p_flops_loss_factor * p_flops_loss
            if self.args.local_rank > -1:
                flops_loss = flops_loss * dist.get_world_size()
            loss = rank_loss + flops_loss
            self.maybe_log_flop(q_flops_loss_factor, p_flops_loss_factor, q_flops_loss, p_flops_loss, rank_loss, flops_loss)
            return (loss, (query_embeds, doc_embeds, neg_doc_embeds)) if return_outputs else loss
            
    def compute_inbatch_contrastive_loss(self, query_embeds, doc_embeds, qids, docids):  
        labels = torch.arange(len(query_embeds), dtype=torch.long, device=query_embeds.device)
        all_doc_embeds = doc_embeds
        all_docids = docids
        negative_mask = self._compute_negative_mask(qids, all_docids)

        similarities = torch.matmul(query_embeds, all_doc_embeds.transpose(0, 1))
        similarities = similarities * self.args.inv_temperature
        similarities = similarities - 10000.0 * negative_mask
        contrast_loss = F.cross_entropy(similarities, labels) 
        if self.args.local_rank > -1:
            contrast_loss = contrast_loss * dist.get_world_size()
        return contrast_loss

    def compute_contrastive_loss(self, query_embeds, doc_embeds, neg_doc_embeds, qids, docids, neg_docids):  
        labels = torch.arange(len(query_embeds), dtype=torch.long, device=query_embeds.device)
        all_doc_embeds = torch.vstack((doc_embeds, neg_doc_embeds))
        all_docids = torch.hstack((docids, neg_docids))
        negative_mask = self._compute_negative_mask(qids, all_docids)

        similarities = torch.matmul(query_embeds, all_doc_embeds.transpose(0, 1))
        similarities = similarities * self.args.inv_temperature
        similarities = similarities - 10000.0 * negative_mask
        contrast_loss = F.cross_entropy(similarities, labels) 
        if self.args.local_rank > -1:
            contrast_loss = contrast_loss * dist.get_world_size()
        return contrast_loss

    @torch.no_grad()
    def _compute_negative_mask(self, qids, docids):
        negative_mask = torch.zeros((len(qids), len(docids)), dtype=torch.bool, device=qids.device)
        for i, qid in enumerate(qids):
            for d in self.qrels[qid.item()]:
                negative_mask[i] = torch.logical_or(negative_mask[i], docids==d)
        negative_mask = negative_mask.type(torch.float32)
        negative_mask.fill_diagonal_(0)
        return negative_mask

    def _gather_tensor(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.args.local_rank] = t
        all_tensors = torch.cat(all_tensors)
        return all_tensors

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        return 0


class BackboneContrastSpladeFinetuner(BaseContrastSpladeFinetuner, Trainer):
    def __init__(self, qrels, *args, **kwargs):
        super(BaseContrastSpladeFinetuner, self).__init__(*args, **kwargs)
        self.qrels = qrels # is used to compute negative mask


class AdapterContrastSpladeFinetuner(BaseContrastSpladeFinetuner, AdapterTrainer):
    def __init__(self, qrels, *args, **kwargs):
        super(AdapterContrastSpladeFinetuner, self).__init__(*args, **kwargs)
        self.qrels = qrels # is used to compute negative mask


