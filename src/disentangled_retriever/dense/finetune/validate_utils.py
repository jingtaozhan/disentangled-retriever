import torch
import logging
import pytrec_eval
from typing import Dict
from collections import defaultdict
from transformers.trainer_utils import is_main_process

from ..evaluate.index_utils import (
    load_corpus, load_queries,
    encode_dense_corpus, encode_dense_query, batch_dense_search, create_index
    )
from disentangled_retriever.evaluate import pytrec_evaluate

logger = logging.getLogger(__name__)


def load_validation_set(corpus_path, query_path, qrel_path):
    
    with open(qrel_path, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
    corpus = load_corpus(corpus_path)
    queries = load_queries(query_path)
    return corpus, queries, qrel


def validate_during_training(trainer, eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix: str = "eval",) -> Dict[str, float]:
    torch.cuda.empty_cache()
    corpus, queries, qrels = trainer.eval_dataset
    fp16, bf16 = trainer.args.fp16, trainer.args.bf16
    trainer.args.fp16, trainer.args.bf16 = False, False
    dataloader_drop_last = trainer.args.dataloader_drop_last
    trainer.args.dataloader_drop_last = False
    corpus_embeds, corpus_ids = encode_dense_corpus(corpus, trainer.model, trainer.tokenizer, trainer.args, 1, return_embeds=True, verbose=is_main_process(trainer.args.local_rank))
    query_embeds, query_ids = encode_dense_query(queries, trainer.model, trainer.tokenizer, trainer.args)
    trainer.args.fp16, trainer.args.bf16 = fp16, bf16
    trainer.args.dataloader_drop_last = dataloader_drop_last

    torch.cuda.empty_cache()
    index = create_index(corpus_embeds, 0 if trainer.args.local_rank < 0 else trainer.args.local_rank)
    all_topk_scores, all_topk_ids = batch_dense_search(
        query_ids, query_embeds,
        corpus_ids, index, 
        topk=10, 
        batch_size=512)

    run_results = defaultdict(dict)
    for qid, topk_scores, topk_ids in zip(query_ids, all_topk_scores, all_topk_ids):
        for i, (score, docid) in enumerate(zip(topk_scores, topk_ids)):
            run_results[qid.item()][docid.item()] = score.item()
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