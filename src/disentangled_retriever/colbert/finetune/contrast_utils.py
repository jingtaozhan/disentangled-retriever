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


class BaseContrastColBERTFinetuner:
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute contrastive loss.
        """
        query_embeds = model(**inputs['query_input']) # Nq, dim
        doc_embeds = model(**inputs['doc_input']) # Nq, dim
        qids = self._prepare_input(inputs['qids']).contiguous() # _prepare_input to gpu
        docids = self._prepare_input(inputs['docids']).contiguous()
        
        if self.args.local_rank > -1:
            query_embeds = self._gather_tensor(query_embeds)
            doc_embeds = self._gather_tensor(doc_embeds)
            qids, docids = self._gather_tensor(qids), self._gather_tensor(docids) 

        if 'neg_doc_input' not in inputs:
            loss = self.compute_inbatch_contrastive_loss(query_embeds, doc_embeds, qids, docids)
            return (loss, (query_embeds, doc_embeds)) if return_outputs else loss
        else:
            neg_doc_embeds = model(**inputs['neg_doc_input']) 
            neg_docids = self._prepare_input(inputs['neg_docids']).contiguous()
            if self.args.local_rank > -1:
                neg_doc_embeds = self._gather_tensor(neg_doc_embeds)
                neg_docids = self._gather_tensor(neg_docids)
            loss = self.compute_contrastive_loss(query_embeds, doc_embeds, neg_doc_embeds, qids, docids, neg_docids)
            return (loss, (query_embeds, doc_embeds, neg_doc_embeds)) if return_outputs else loss

    def compute_inbatch_contrastive_loss(self, query_embeds, doc_embeds, qids, docids):  
        labels = torch.arange(len(query_embeds), dtype=torch.long, device=query_embeds.device)
        all_doc_embeds = doc_embeds
        all_docids = docids
        negative_mask = self._compute_negative_mask(qids, all_docids)

        similarities = self.model.compute_similarity(query_embeds, all_doc_embeds)
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

        similarities = self.model.compute_similarity(query_embeds, all_doc_embeds)
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


class BackboneContrastColBERTFinetuner(BaseContrastColBERTFinetuner, Trainer):
    def __init__(self, qrels, *args, **kwargs):
        super(BackboneContrastColBERTFinetuner, self).__init__(*args, **kwargs)
        self.qrels = qrels # is used to compute negative mask


class AdapterContrastColBERTFinetuner(BaseContrastColBERTFinetuner, AdapterTrainer):
    def __init__(self, qrels, *args, **kwargs):
        super(AdapterContrastColBERTFinetuner, self).__init__(*args, **kwargs)
        self.qrels = qrels # is used to compute negative mask

    def _load_from_checkpoint(self, resume_from_checkpoint):
        super()._load_from_checkpoint(resume_from_checkpoint)
        self._move_model_to_device(self.model, self.args.device)

