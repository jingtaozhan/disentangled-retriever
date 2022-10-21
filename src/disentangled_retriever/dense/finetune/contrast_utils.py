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

from .validate_utils import validate_during_training

logger = logging.getLogger(__name__)


@dataclass
class FinetuneCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_query_len: int, max_doc_len: int, padding=True):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.padding = padding

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # tokenizing batch of text is much faster
        query_input = self.tokenizer(
            [x['query'] for x in features],
            padding=self.padding,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True,
            max_length=self.max_query_len
        )
        query_input['position_ids'] = torch.arange(0, query_input['input_ids'].size(1))[None, :]
        doc_input = self.tokenizer(
            [x['doc'] for x in features],
            padding=self.padding,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True,
            max_length=self.max_doc_len
        )
        doc_input['position_ids'] = torch.arange(0, doc_input['input_ids'].size(1))[None, :]
        # we have to prevent inbatch false negatives when gathering tensors in the trainer
        # because each distributed process has its own collators
        qids = torch.tensor([x['qid'] for x in features], dtype=torch.long)
        docids = torch.tensor([x['docid'] for x in features], dtype=torch.long)

        batch_data = {
                "query_input": query_input,
                "doc_input": doc_input,
                "qids": qids,
                "docids": docids,
            }

        if 'neg_docs' in features[0]:
            neg_doc_input = self.tokenizer(
                sum([x['neg_docs'] for x in features], []),
                padding=self.padding,
                return_tensors='pt',
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                truncation=True,
                max_length=self.max_doc_len
            )     
            neg_doc_input['position_ids'] = torch.arange(0, neg_doc_input['input_ids'].size(1))[None, :]
            neg_docids = torch.tensor(sum([x['neg_docids'] for x in features], []), dtype=torch.long)  
            batch_data.update({
                "neg_doc_input": neg_doc_input,
                "neg_docids": neg_docids,
            }) 
        return batch_data


class QDRelDataset(Dataset):
    def __init__(self, 
            tokenizer: PreTrainedTokenizer, 
            qrel_path: str, 
            query_path: str, 
            corpus_path: str, 
            max_query_len: int, 
            max_doc_len: int, 
            negative: str, 
            neg_per_query: int,
            rel_threshold=1, 
            verbose=True):
        '''
        negative: choices from `random' or a path to a json file that contains \
            the qid:neg_pid_lst  
        '''
        super().__init__()
        self.tokenizer = tokenizer
        self.queries, qid2offset = [], dict()
        for idx, line in enumerate(tqdm(open(query_path), disable=not verbose, mininterval=10)):
            qid, query = line.split("\t")
            qid2offset[qid] = idx
            self.queries.append(query.strip())

        self.corpus, docid2offset = [], dict()
        for idx, line in enumerate(tqdm(open(corpus_path), disable=not verbose, mininterval=10)):
            splits = line.split("\t")
            if len(splits) == 2:
                docid, body = splits
            else:
                raise NotImplementedError()
            docid2offset[docid] = idx
            self.corpus.append(body.strip())

        self.qrels = defaultdict(list)
        for line in tqdm(open(qrel_path), disable=not verbose, mininterval=10):
            qid, _, docid, rel = line.split()
            if int(rel) >= rel_threshold:
                qoffset = qid2offset[qid]
                docoffset = docid2offset[docid]
                self.qrels[qoffset].append(docoffset)
        
        if os.path.exists(negative):
            self.negative = {}
            for line in tqdm(open(negative), disable=not verbose, mininterval=10, desc="read negatives"):
                qid, neg_docids = line.strip().split("\t")
                neg_docids = neg_docids.split(" ")
                qoffset, neg_docoffsets = qid2offset[qid], [docid2offset[docid] for docid in neg_docids]
                assert len(set(neg_docoffsets) & set(self.qrels[qoffset])) == 0, "Negative docids and relevant docids should not overlap."
                self.negative[qoffset] = neg_docoffsets
            assert set(self.negative.keys()) == set(self.qrels.keys())
        else:
            self.negative = negative
        self.neg_per_query = neg_per_query

        self.qids = sorted(self.qrels.keys())
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.qrels = dict(self.qrels)

    def get_qrels(self):
        return self.qrels

    def __len__(self):
        return len(self.qids)
    
    def __getitem__(self, index):
        '''
        We do not tokenize text here and instead tokenize batch of text in the collator because
            a. Tokenizing batch of text is much faster then tokenizing one by one
            b. Usually, the corpus is too large and we cannot afford to use multiple num workers
        '''
        qid = self.qids[index]
        query = self.queries[qid]
        rel_docids = self.qrels[qid]
        docid = random.choice(rel_docids)
        doc = self.corpus[docid]
        data = {
            "query": query,
            "doc": doc,
            "docid": docid,
            "qid": qid
        }
        if self.neg_per_query > 0:
            if self.negative == "random":
                neg_docids = random.sample(range(len(self.corpus)), self.neg_per_query)
            elif isinstance(self.negative, Dict):
                neg_docids = random.sample(self.negative[qid], self.neg_per_query)
            else:
                raise NotImplementedError()
            neg_docs = [self.corpus[neg_docid] for neg_docid in neg_docids ]
            data.update({"neg_docids": neg_docids, "neg_docs": neg_docs})
        return data


class BaseContrastDenseFinetuner:
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

    def evaluate(self, 
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",) -> Dict[str, float]:
        metrics = validate_during_training(self, eval_dataset, ignore_keys, metric_key_prefix)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


class BackboneContrastDenseFinetuner(BaseContrastDenseFinetuner, Trainer):
    def __init__(self, qrels, *args, **kwargs):
        super(BackboneContrastDenseFinetuner, self).__init__(*args, **kwargs)
        self.qrels = qrels # is used to compute negative mask


class AdapterContrastDenseFinetuner(BaseContrastDenseFinetuner, AdapterTrainer):
    def __init__(self, qrels, *args, **kwargs):
        super(AdapterContrastDenseFinetuner, self).__init__(*args, **kwargs)
        self.qrels = qrels # is used to compute negative mask


