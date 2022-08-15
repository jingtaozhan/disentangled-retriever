import re
import math
import torch
import faiss
import logging
import numpy as np
from tqdm import tqdm
import faiss.contrib.torch_utils
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import TrainingArguments, Trainer

logger = logging.getLogger(__name__)


def load_corpus(corpus_path, verbose=True):
    corpus = {}
    for line in tqdm(open(corpus_path), mininterval=10, disable=not verbose):
        splits = line.split("\t")
        if len(splits) == 2:
            corpus_id, text = splits
            corpus[corpus_id] = text.strip()
        else:  
            raise NotImplementedError()
    return corpus


def load_queries(query_path):
    queries = {}
    for line in open(query_path):
        qid, text = line.split("\t")
        queries[qid] = text
    return queries
    

class TextDataset(Dataset):
    def __init__(self, text_lst: List[str], text_ids: List[int]=None):
        self.text_lst = text_lst
        self.text_ids = text_ids
        assert self.text_ids is None or len(text_lst) == len(text_ids)
    
    def __len__(self):
        return len(self.text_lst)

    def __getitem__(self, item):
        if self.text_ids is not None:
            return self.text_ids[item], self.text_lst[item]
        else:
            return self.text_lst[item]


def get_collator_func(tokenizer, max_length):
    def collator_fn(batch):
        if isinstance(batch[0], tuple):
            ids = torch.LongTensor([x[0] for x in batch])
            features = tokenizer([x[1] for x in batch], padding=True, truncation=True, max_length=max_length)
            return {
                'input_ids': torch.LongTensor(features['input_ids']),
                'attention_mask': torch.LongTensor(features['attention_mask']),
                'text_ids': ids,
            }
        else:
            assert isinstance(batch[0], str)
            features = tokenizer(batch, padding=True, truncation=True, max_length=max_length)
            return {
                'input_ids': torch.LongTensor(features['input_ids']),
                'attention_mask': torch.LongTensor(features['attention_mask']),
            }
    return collator_fn


class Evaluater(Trainer):
    def prediction_step(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert prediction_loss_only == False
        assert ignore_keys is None
        inputs = self._prepare_inputs(inputs)
        text_ids = inputs['text_ids']
        del inputs['text_ids']

        with torch.no_grad():
            loss = None
            with self.autocast_smart_context_manager():
                logits = model(**inputs).detach()
        return (loss, logits, text_ids)


class DenseEvaluater(Trainer):
    def prediction_step(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert prediction_loss_only == False
        assert ignore_keys is None
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = None
            with self.autocast_smart_context_manager():
                logits = model(**inputs).detach().contiguous()
        return (loss, logits, None)


def encode_dense_corpus(corpus: Dict[Union[str, int], str], model, 
        tokenizer, eval_args: TrainingArguments, split_corpus_num=1, return_embeds=True, verbose=True):
    '''
    Avoid out-of-memory error (dense embeds are memory-inefficient)
    return_embeds can be used in distributed training to save memory
    '''
    logger.info("Sorting Corpus by document length (Longest first)...")
    if re.search("[\u4e00-\u9FFF]", list(corpus.values())[0]):
        logger.info("Automatically detect the corpus is in chinese and will use len(str) to sort the corpus for efficiently encoding")
        corpus_ids = np.array(sorted(corpus, key=lambda k: len(corpus[k]), reverse=True))
    else:
        logger.info("Use len(str.split()) to sort the corpus for efficiently encoding")
        corpus_ids = np.array(sorted(corpus, key=lambda k: len(corpus[k].split()), reverse=True))
    if return_embeds:
        corpus_embeds = np.empty((len(corpus_ids), model.config.hidden_size), dtype=np.float32)
        write_num = 0
    else:
        corpus_embeds = None
    for doc_ids in tqdm(np.array_split(corpus_ids, split_corpus_num), 
            disable=not verbose, desc="Split corpus encoding"):
        doc_text = [corpus[did] for did in doc_ids]
        doc_dataset = TextDataset(
            doc_text, 
        )
        doc_out = DenseEvaluater(
            model=model,
            args=eval_args,
            data_collator=get_collator_func(tokenizer, 512),
            tokenizer=tokenizer,
        ).predict(doc_dataset)
        if return_embeds:
            doc_embeds = doc_out.predictions
            assert len(doc_embeds) == len(doc_text)
            corpus_embeds[write_num:write_num+len(doc_embeds)] = doc_embeds
            write_num += len(doc_embeds)
    return corpus_embeds, corpus_ids


def encode_dense_query(queries: Dict[Union[str, int], str], model, tokenizer, eval_args: TrainingArguments):
    logger.info("Encoding Queries...")
    query_ids = sorted(list(queries.keys()))
    queries_text = [queries[qid] for qid in query_ids]
    query_dataset = TextDataset(queries_text)
    query_out = DenseEvaluater(
        model=model,
        args=eval_args,
        data_collator=get_collator_func(tokenizer, 512),
        tokenizer=tokenizer,
    ).predict(query_dataset)
    query_embeds = query_out.predictions
    assert len(query_embeds) == len(queries_text)
    return query_embeds, np.array(query_ids)


def dense_search(query_ids: np.ndarray, query_embeds:np.ndarray, 
    corpus_ids: np.ndarray, index: faiss.IndexFlatIP, topk: int):
    topk_scores, topk_idx = index.search(query_embeds, topk)
    topk_ids = np.vstack([corpus_ids[x] for x in topk_idx])
    assert len(query_ids) == len(topk_scores) == len(topk_ids)
    return topk_scores, topk_ids
  

def batch_dense_search(query_ids: np.ndarray, query_embeds:np.ndarray, 
    corpus_ids: np.ndarray, index: faiss.IndexFlatIP, 
    topk: int, batch_size: int):

    all_topk_scores, all_topk_ids = [], []
    iterations = math.ceil(len(query_ids) / batch_size)
    for query_id_iter, query_embeds_iter in tqdm(zip(
        np.array_split(query_ids, iterations), 
        np.array_split(query_embeds, iterations),
    ), total=iterations, desc="Batch search"):
        topk_scores, topk_ids = dense_search(
            query_id_iter, query_embeds_iter,
            corpus_ids, index, topk
        )
        all_topk_scores.append(topk_scores)
        all_topk_ids.append(topk_ids)
    all_topk_scores = np.concatenate(all_topk_scores, axis=0)
    all_topk_ids = np.concatenate(all_topk_ids, axis=0)
    return all_topk_scores, all_topk_ids


def create_index(corpus_embeds: np.ndarray, single_gpu_id=None):
    index = faiss.IndexFlatIP(corpus_embeds.shape[1])
    if faiss.get_num_gpus() == 1 or single_gpu_id is not None:
        res = faiss.StandardGpuResources()  # use a single GPU
        res.setTempMemory(128*1024*1024)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = False
        if single_gpu_id is None:
            single_gpu_id = 0
        index = faiss.index_cpu_to_gpu(res, single_gpu_id, index, co)
    else:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = False
        index = faiss.index_cpu_to_all_gpus(index, co)
    index.add(corpus_embeds)
    return index


def concat_title_body(doc: Dict[str, str]):
    body = doc['text'].strip()
    if "title" in doc and len(doc['title'].strip())> 0:
        title = doc['title'].strip()
        if title[-1] in "!.?。！？":
            text = title + " " + body
        else:
            text = title + ". " + body
    else:
        text = body
    return text