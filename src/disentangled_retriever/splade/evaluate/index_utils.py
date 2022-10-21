import re
import os
import sys
import json
import math
import torch
import faiss
import queue
import logging
import threading
import subprocess
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import faiss.contrib.torch_utils
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any, Union

import transformers
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import is_main_process

from ...dense.evaluate.index_utils import load_corpus, load_queries, TextDataset, get_collator_func

logger = logging.getLogger(__name__)


class SpladeEvaluater(Trainer):
    rounding_func = torch.round

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
                logits = self.rounding_func(logits * 100).type(torch.int32)
        return (loss, logits, None)


def encode_sparse_text(
        evaluator_cls: Trainer,
        corpus: Dict[Union[str, int], str], 
        model, 
        tokenizer, 
        eval_args: TrainingArguments, 
        encode_is_qry,
        encoded_save_path,
        split_corpus_num=1, 
        verbose=True
    ):
    saver_queue = queue.Queue(maxsize=3)
    def _saver_thread():
        if encode_is_qry:
            save_path = f"{encoded_save_path}.{eval_args.local_rank}"
        else:
            save_path = f"{encoded_save_path}/corpus.jsonl.{eval_args.local_rank}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        collection_file = open(save_path, "w")
        for lines in iter(saver_queue.get, None):
            for line in lines:
                collection_file.write(line)
        collection_file.close()

    thread = threading.Thread(target=_saver_thread)
    thread.start()

    corpus_ids = np.array(list(corpus.keys()))
    if eval_args.local_rank >= 0:
        corpus_ids = np.array_split(corpus_ids, eval_args.world_size)[eval_args.local_rank]

    logger.info("Sorting Corpus by document length (Longest first)...")
    if re.search("[\u4e00-\u9FFF]", list(corpus.values())[0]):
        logger.info("Automatically detect the corpus is in chinese and will use len(str) to sort the corpus for efficiently encoding")
        corpus_ids = np.array(sorted(corpus_ids.tolist(), key=lambda k: len(corpus[k]), reverse=True))
    else:
        logger.info("Use len(str.split()) to sort the corpus for efficiently encoding")
        corpus_ids = np.array(sorted(corpus_ids.tolist(), key=lambda k: len(corpus[k].split()), reverse=True))
    
    vocab_dict = tokenizer.get_vocab()
    vocab_dict = {v: k for k, v in vocab_dict.items()}

    for text_ids in tqdm(np.array_split(corpus_ids, split_corpus_num), 
            disable=not verbose, desc="encoding"):

        local_rank = eval_args.local_rank
        eval_args.disable_tqdm = True # not is_main_process(local_rank)
        eval_args.local_rank = -1 # so that trainer will not gather tensor
        evaluator = evaluator_cls(
            model=model,
            args=eval_args,
            data_collator=get_collator_func(tokenizer, 512),
            tokenizer=tokenizer,
        )
        transformers.utils.logging.set_verbosity(transformers.logging.WARNING)
        outputs = evaluator.predict(TextDataset([corpus[_id] for _id in text_ids]))
        eval_args.local_rank = local_rank

        batch_output = []
        for rep, id_ in zip(outputs.predictions, text_ids):
            idx = np.nonzero(rep)
            data = rep[idx]
            dict_splade = dict()
            for id_token, value_token in zip(idx[0],data):
                if value_token > 0:
                    real_token = vocab_dict[id_token]
                    dict_splade[real_token] = int(value_token)
            if len(dict_splade.keys()) == 0:
                print("empty input =>", id_)
                dict_splade[vocab_dict[998]] = 1  # in case of empty doc we fill with "[unused993]" token (just to fill
                # and avoid issues with anserini), in practice happens just a few times ...
            if not encode_is_qry:
                dict_doc = dict(id=id_, content="", vector=dict_splade)
                json_dict = json.dumps(dict_doc)  
                batch_output.append(json_dict + "\n")
            else:
                if len(dict_splade) > 1024: # maxClauseCount limitation in Anserini
                    dict_splade = dict(sorted(list(dict_splade.items()), key=lambda x: x[1], reverse=True)[:1024])
                string_splade = " ".join(
                    [" ".join([str(real_token)] * freq) for real_token, freq in dict_splade.items()])
                batch_output.append(str(id_) + "\t" + string_splade + "\n")
        saver_queue.put(batch_output)

        if eval_args.local_rank > -1:
            dist.barrier()

    saver_queue.put(None)
    print("#> Joining saver thread.")
    thread.join()

    if encode_is_qry and eval_args.local_rank == 0:
        with open(encoded_save_path, 'w') as outfile:
            for fname in list(f"{encoded_save_path}.{i}" for i in range(eval_args.world_size)):
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)


def encode_splade_text(
        corpus: Dict[Union[str, int], str], 
        model, 
        tokenizer, 
        eval_args: TrainingArguments, 
        encode_is_qry,
        encoded_save_path,
        split_corpus_num=1, 
        verbose=True
    ):
    return encode_sparse_text(
        SpladeEvaluater,
        corpus, 
        model, 
        tokenizer, 
        eval_args, 
        encode_is_qry,
        encoded_save_path,
        split_corpus_num=split_corpus_num, 
        verbose=verbose
    )


assert "ANSERINI_ROOT" in os.environ
ANSERINI_ROOT = os.environ["ANSERINI_ROOT"]

def batch_anserini_search(query_path, index_path, output_path, topk, batch_size):
    
    subprocess.check_call([
        "bash", os.path.join(ANSERINI_ROOT, "target/appassembler/bin/SearchCollection"),
        "-hits", str(topk), "-parallelism", str(batch_size),
        "-index", index_path, 
        "-topicreader", "TsvInt", "-topics", query_path,
        "-output", output_path, "-format", "trec",
        "-impact", "-pretokenized"
    ])


def create_anserini_index(input_path, output_path, threads=16):
    subprocess.check_call([
        "bash", os.path.join(ANSERINI_ROOT, "target/appassembler/bin/IndexCollection"),
        "-collection", "JsonVectorCollection",
        "-input", input_path,
        "-index", output_path,
        "-generator", "DefaultLuceneDocumentGenerator", 
        "-impact", "-pretokenized", "-threads", str(threads)
    ])
    return output_path
