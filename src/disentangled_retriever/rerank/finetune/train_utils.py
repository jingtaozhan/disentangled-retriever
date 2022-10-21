import os
import torch
import random
import logging
import pytrec_eval
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Any, Union, Optional, Tuple
from torch.utils.data import Dataset
from transformers import (
    AdapterTrainer,
    PreTrainedTokenizer,
    Trainer
)

from ...evaluate import pytrec_evaluate
from ..evaluate.eval_utils import load_corpus, load_queries, load_candidates, rerank


logger = logging.getLogger(__name__)


@dataclass
class RerankFinetuneCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, features: List[List[Tuple[str]]]) -> Dict[str, Any]:
        # tokenizing batch of text is much faster
        features = sum(features, [])
        input_dict = self.tokenizer(
            features,
            padding=True,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True,
            truncation=True,
            max_length=self.max_seq_len
        )
        return input_dict


class TrainRerankDataset(Dataset):
    def __init__(self, 
            tokenizer: PreTrainedTokenizer, 
            query_path: str, 
            corpus_path: str, 
            qrel_path: str,
            candidate_path: str,
            neg_per_query,
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
        self.qrels = dict(self.qrels)

        self.negs = dict()
        if candidate_path is not None:
            for line in tqdm(open(candidate_path), disable=not verbose, mininterval=10, desc="load negs"):
                qid, pids = line.split("\t")
                pids = pids.strip().split()
                qoffset, poffsets = qid2offset[qid], [docid2offset[pid] for pid in pids]
                if qoffset in self.qrels:
                    poffsets = list(set(filter(lambda x: x not in self.qrels[qoffset], poffsets)))
                    self.negs[qoffset] = poffsets
            # else:
            #     logger.info(f"Query:{qid} in negs but not in qrels.")

        logger.info(f"{len(self.negs)} queries have negatives")
        logger.info(f"{len(self.qrels) - len(self.negs)} queries do not have negatives and will use random negatives.")
        self.qids = sorted(self.qrels.keys())
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
        if qid not in self.negs:
            # logger.info(f"Missing qid in negs: {qid}")
            neg_docids = random.sample(range(len(self.corpus)), self.neg_per_query)
        else:
            candidate_negs = self.negs[qid]
            if len(candidate_negs) >= self.neg_per_query:
                neg_docids = random.sample(candidate_negs, self.neg_per_query)
            else:
                neg_docids = candidate_negs + random.sample(range(len(self.corpus)), self.neg_per_query-len(candidate_negs))
        data = [(query, self.corpus[docid]) for docid in [rel_docid]+neg_docids]
        return data


def load_validation_set(corpus_path, query_path, candidate_path, qrel_path):
    
    with open(qrel_path, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
    corpus = load_corpus(corpus_path)
    queries = load_queries(query_path)
    candidates = load_candidates(candidate_path)
    return corpus, queries, candidates, qrel


def validate_during_training(trainer, eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix: str = "eval",) -> Dict[str, float]:
    torch.cuda.empty_cache()
    corpus, queries, candidates, qrels = trainer.eval_dataset
    fp16, bf16 = trainer.args.fp16, trainer.args.bf16
    trainer.args.fp16, trainer.args.bf16 = False, False
    dataloader_drop_last = trainer.args.dataloader_drop_last
    trainer.args.dataloader_drop_last = False
    scores_dict = rerank(
        queries = queries,
        corpus = corpus,
        candidates = candidates,
        model = trainer.model, 
        tokenizer = trainer.tokenizer, 
        eval_args = trainer.args
    )
    trainer.args.fp16, trainer.args.bf16 = fp16, bf16
    trainer.args.dataloader_drop_last = dataloader_drop_last

    run_results = defaultdict(dict)

    for qid, pid_list in scores_dict.items():
        for i, (docid, score) in enumerate(pid_list):
            run_results[qid][docid] = score
    metrics = {}
    for category, cat_metrics in pytrec_evaluate(
            qrels, 
            dict(run_results), 
            k_values =(10, ),
            mrr_k_values = (10, ),).items():
        if category == "perquery":
            continue
        for metric, score in cat_metrics.items():
            metrics[f"{metric_key_prefix}_{metric}"] = score
    torch.cuda.empty_cache()
    trainer.log(metrics)
    
    return metrics


class BaseRerankFinetuner:

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute contrastive loss.
        """
        scores = model(**inputs).logits.reshape(-1, 1 + self.args.neg_per_query) 
        labels = torch.zeros((len(scores), ), dtype=torch.long, device=scores.device)
        loss = F.cross_entropy(scores, labels) 
        return (loss, scores) if return_outputs else loss

    def evaluate(self, 
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",) -> Dict[str, float]:
        metrics = validate_during_training(self, eval_dataset, ignore_keys, metric_key_prefix)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics
    
    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        return 0


class BackboneRerankFinetuner(BaseRerankFinetuner, Trainer):
    pass


class AdapterRerankFinetuner(BaseRerankFinetuner, AdapterTrainer):

    def _load_from_checkpoint(self, resume_from_checkpoint):
        super()._load_from_checkpoint(resume_from_checkpoint)
        if not self.args.not_train_embedding:
            embedding_loaded = False
            for file_name in os.listdir(resume_from_checkpoint):
                dir_path = os.path.join(resume_from_checkpoint, file_name)
                if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                    if "," not in file_name and "adapter_config.json" in os.listdir(dir_path) and "embeddings.bin" in os.listdir(dir_path):
                        self.model.base_model.embeddings.load_state_dict(
                            torch.load(os.path.join(dir_path, "embeddings.bin"), map_location="cpu")) 
                        assert embedding_loaded == False, "Multiple embeddings.bin exists."
                        embedding_loaded = True
            assert embedding_loaded, embedding_loaded
        self._move_model_to_device(self.model, self.args.device)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir, state_dict)
        if not self.args.not_train_embedding:
            for adapter_name in self.model.config.adapters:
                torch.save(self.model.base_model.embeddings.state_dict(), 
                    os.path.join(output_dir, adapter_name, "embeddings.bin"))
