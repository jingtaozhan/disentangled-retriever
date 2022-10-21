import re
import math
import torch
import faiss
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import faiss.contrib.torch_utils
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import BertTokenizer, TrainingArguments, Trainer


logger = logging.getLogger(__name__)


def load_candidates(candidate_path, verbose=True, topk=None):
    candidates = defaultdict(list)
    for line in tqdm(open(candidate_path), mininterval=10, disable=not verbose):
        splits = line.strip().split()
        if len(splits) == 3: # msmarco style
            qid, pid, rank = splits
        elif len(splits) == 6: # trec style
            assert splits[1] == "Q0", line
            qid, _, pid, rank= splits[:4]
        else: # msmarco doc 
            raise NotImplementedError(f"line: {line}")
        rank = int(rank)
        if topk is None or rank <= topk:
            candidates[qid].append(pid)
    return candidates


def load_corpus(corpus_path, verbose=True):
    corpus = {}
    for line in tqdm(open(corpus_path), mininterval=10, disable=not verbose):
        splits = line.split("\t")
        if len(splits) == 2: # msmarco passage
            corpus_id, text = splits
            corpus[corpus_id] = text.strip()
        else: # msmarco doc 
            raise NotImplementedError()
    return corpus


def load_queries(query_path):
    queries = {}
    for line in open(query_path):
        qid, text = line.split("\t")
        queries[qid] = text
    return queries
    

class EvalRerankDataset(Dataset):
    def __init__(self, 
        queries: Dict[Union[str, int], str], 
        corpus: Dict[Union[str, int], str],
        pair_ids: List[int]=None):

        self.queries = queries
        self.corpus = corpus
        self.pair_ids = pair_ids
    
    def __len__(self):
        return len(self.pair_ids)

    def __getitem__(self, item):
        qid, pid = self.pair_ids[item]
        return (self.queries[qid], self.corpus[pid])


def get_collator_func(tokenizer: BertTokenizer, max_length: int):
    def collator_fn(batch):
        features = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors='pt'
        )
        return features
    return collator_fn


class RerankEvaluater(Trainer):
    def prediction_step(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert prediction_loss_only == False
        assert ignore_keys is None
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = None
            with self.autocast_smart_context_manager():
                logits = model(**inputs).logits.detach().contiguous()
        return (loss, logits, None)


def rerank(
        queries: Dict[Union[str, int], str], 
        corpus: Dict[Union[str, int], str],
        candidates: Dict[Union[str, int], List[Union[str, int]]],
        model, 
        tokenizer, 
        eval_args: TrainingArguments):
    logger.info("Encoding Queries...")
    pair_ids = []
    for qid, pids in candidates.items():
        for pid in pids:
            pair_ids.append((qid, pid))

    if re.search("[\u4e00-\u9FFF]", list(corpus.values())[0]):
        logger.info("Automatically detect the corpus is in chinese and will use len(str) to sort the corpus for efficiently encoding")
        pair_ids = sorted(pair_ids, key=lambda x: len(queries[x[0]])+len(corpus[x[1]]), reverse=True)
    else:
        logger.info("Use len(str.split()) to sort the corpus for efficiently encoding")
        pair_ids = sorted(pair_ids, key=lambda x: len(queries[x[0]].split())+len(corpus[x[1]].split()), reverse=True)

    dataset = EvalRerankDataset(queries, corpus, pair_ids)
    output = RerankEvaluater(
        model=model,
        args=eval_args,
        data_collator=get_collator_func(tokenizer, 512),
        tokenizer=tokenizer,
    ).predict(dataset)
    scores = output.predictions.reshape(-1)
    assert len(scores) == len(pair_ids), f"{output.predictions.shape} {scores.shape}, {len(pair_ids)}"
    ret_dict = defaultdict(list)
    for (qid, pid), s in zip(pair_ids, scores):
        ret_dict[qid].append((pid, s.item()))
    ret_dict = {
        qid: sorted(pid_lst, key=lambda x: x[1], reverse=True) for qid, pid_lst in ret_dict.items()
    }
    return ret_dict