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

from .validate_utils import validate_during_training


logger = logging.getLogger(__name__)


@dataclass
class FinetuneCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_query_len: int, max_doc_len: int, ):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # tokenizing batch of text is much faster
        query_input = self.tokenizer(
            [x['query'] for x in features],
            padding=True,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True,
            max_length=self.max_query_len
        )
        query_input['position_ids'] = torch.arange(0, query_input['input_ids'].size(1))[None, :]
        # the first is positive
        # the other neg_per_query is negative
        # multiply the number of queries
        doc_input = self.tokenizer(
            sum((x['docs'] for x in features), []),
            padding=True,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True,
            max_length=self.max_doc_len
        )
        # in distributed training, without passing position ids and separately encoding queries/docs cause inplace-modification errors
        doc_input['position_ids'] = torch.arange(0, doc_input['input_ids'].size(1))[None, :]
        # equals the number of docs
        ce_scores = torch.tensor([x['ce_scores'] for x in features])
        batch_data = {
                "query_input": query_input,
                "doc_input": doc_input,
                "ce_scores": ce_scores,
        }
        return batch_data


class QDRelDataset(Dataset):
    def __init__(self, 
            tokenizer: PreTrainedTokenizer, 
            query_path: str, 
            corpus_path: str, 
            max_query_len: int, 
            max_doc_len: int, 
            qrel_path: str,
            ce_scores_file: str,
            neg_per_query: int,
            verbose=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.queries, qid2offset = [], dict()
        for idx, line in enumerate(tqdm(open(query_path), disable=not verbose, mininterval=10, desc="load queries")):
            qid, query = line.split("\t")
            qid2offset[qid] = idx
            self.queries.append(query.strip())

        self.corpus, docid2offset = [], dict()
        for idx, line in enumerate(tqdm(open(corpus_path), disable=not verbose, mininterval=10, desc="load corpus")):
            splits = line.split("\t")
            if len(splits) == 2:
                docid, body = splits
            else:
                raise NotImplementedError()
            docid2offset[docid] = idx
            self.corpus.append(body.strip())

        self.qrels = defaultdict(list)
        for line in tqdm(open(qrel_path), disable=not verbose, mininterval=10, desc="load qrels"):
            qid, _, docid, rel = line.split()
            if int(rel) >= 1:
                qoffset = qid2offset[qid]
                docoffset = docid2offset[docid]
                self.qrels[qoffset].append(docoffset)

        if ce_scores_file.endswith(".gz"):
            with gzip.open(ce_scores_file, 'rb') as fIn:
                load_ce_scores = pickle.load(fIn)
        else:
            with open(ce_scores_file, 'rb') as fIn:
                load_ce_scores = pickle.load(fIn)
        self.ce_scores = {
            qid2offset[str(qid)]: { docid2offset[str(docid)]: score for docid, score in docid2scores.items()}
            for qid, docid2scores in tqdm(load_ce_scores.items(), disable=not verbose, mininterval=10, desc="load ce scores")
        }
        # for quickly debug
        # self.ce_scores = {qid: { i: 1.0 for i in list(range(200)) + rel_docids} for qid, rel_docids in self.qrels.items()}

        self.qids = sorted(self.qrels.keys())
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.qrels = dict(self.qrels)
        self.neg_per_query = neg_per_query

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
        rel_docid = random.choice(rel_docids)
        neg_docids = random.sample(set(self.ce_scores[qid].keys()) - set(rel_docids), self.neg_per_query)
        # neg_docids = random.sample(range(len(self.corpus)), self.neg_per_query) # for debug
        docids = [rel_docid] + neg_docids
        docs = [self.corpus[docid] for docid in docids]
        scores_dict = self.ce_scores[qid]
        scores = [scores_dict[docid] for docid in docids]
        # scores = [0.0 for docid in docids] # for debug
        data = {
            "query": query,
            "docs": docs,
            "ce_scores": scores
        }
        return data


class BaseDistillDenseFinetuner:
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

    def _compute_inbatch_contrast_loss(self, query_embeds, doc_embeds):
        """
            query_embeds: Nq, dim
            doc_embeds: Nq, (1+n), dim
            Ignore top-n negatives because they have been used in distillation loss
        """
        if self.args.local_rank > -1:
            query_embeds = self._gather_tensor(query_embeds)
            doc_embeds = self._gather_tensor(doc_embeds)

        labels = torch.arange(len(query_embeds), dtype=torch.long, device=query_embeds.device) * doc_embeds.size(1)

        similarities = torch.matmul(query_embeds, doc_embeds.reshape(-1, doc_embeds.size(-1)).transpose(-2, -1)) * self.args.inv_temperature
        contrast_loss = F.cross_entropy(similarities, labels) 
        if self.args.local_rank > -1:
            contrast_loss = contrast_loss * dist.get_world_size()
        return contrast_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute Margin MSE loss.
        """
        loss, (query_embeds, doc_embeds, model_scores) = self._compute_distill_loss(model, inputs, True)
        return (loss, (query_embeds, doc_embeds)) if return_outputs else loss

    def evaluate(self, 
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",) -> Dict[str, float]:
        metrics = validate_during_training(self, eval_dataset, ignore_keys, metric_key_prefix)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics
    
    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        return 0


class BackboneDistillDenseFinetuner(BaseDistillDenseFinetuner, Trainer):
    pass


class AdapterDistillDenseFinetuner(BaseDistillDenseFinetuner, AdapterTrainer):
    pass
