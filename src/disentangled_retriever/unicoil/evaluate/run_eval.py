import os
import math
import json
import torch
import shutil
import logging
import numpy as np
import torch.distributed as dist
from dataclasses import field, dataclass

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    HfArgumentParser, 
    TrainingArguments,
    set_seed)
from transformers.trainer_utils import is_main_process


from ..modeling import AutoUnicoilModel
from .index_utils import (
    encode_unicoil_text, 
    batch_anserini_search,
    create_anserini_index,
    load_corpus, 
    load_queries
)
from disentangled_retriever.evaluate import pytrec_evaluate

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    corpus_path: str = field()
    out_corpus_dir: str = field()

    query_path: str = field()
    out_query_dir: str = field()
    qrel_path: str = field(default=None)


@dataclass
class ModelArguments:
    backbone_name_or_path: str = field()
    adapter_name_or_path: str = field(default=None)
    merge_lora: bool = field(default=False)


@dataclass
class EvalArguments(TrainingArguments):
    topk : int = field(default=1000)
    threads: int = field(default=32)


def load_or_encode_query(model, tokenizer, query_path, out_query_dir, data_args, eval_args):
    out_query_path = os.path.join(out_query_dir, "query.jsonl")
    if (not eval_args.overwrite_output_dir) and os.path.exists(out_query_path):
        logger.info("Load pre-computed query representations")
        return out_query_path
    else:
        queries = load_queries(query_path)
        encode_unicoil_text(
            queries, 
            model, 
            tokenizer, 
            eval_args, 
            encode_is_qry=True, 
            encoded_save_path=out_query_path,
            split_corpus_num=math.ceil(len(queries)/100_000),
            verbose=is_main_process(eval_args.local_rank)
        )
        return out_query_path


def load_or_encode_corpus(model, tokenizer, data_args, eval_args):
    encode_dir = os.path.join(data_args.out_corpus_dir, "encoded")
    if eval_args.overwrite_output_dir or not os.path.exists(data_args.out_corpus_dir):
        corpus = load_corpus(data_args.corpus_path, verbose=is_main_process(eval_args.local_rank))
        if os.path.exists(data_args.out_corpus_dir) and len(os.listdir(data_args.out_corpus_dir)) > 0 and is_main_process(eval_args.local_rank):
            shutil.move(data_args.out_corpus_dir, f"{data_args.out_corpus_dir}-cache")
            logger.info(f"Rename old {data_args.out_corpus_dir}")
        if eval_args.local_rank > -1:
            dist.barrier()
        encode_unicoil_text(
            corpus, 
            model, 
            tokenizer, 
            eval_args, 
            encode_is_qry=False,
            encoded_save_path=encode_dir,
            split_corpus_num = math.ceil(len(corpus)/100_000), 
            verbose = is_main_process(eval_args.local_rank)
        )
        if is_main_process(eval_args.local_rank):
            create_anserini_index(encode_dir, data_args.out_corpus_dir, threads=eval_args.threads)
    return data_args.out_corpus_dir


def search_and_compute_metrics(index_path, encoded_query_path, out_metric_path, out_query_dir, qrel_path, eval_args):
    out_run_path = os.path.join(out_query_dir, "run.tsv")
    batch_anserini_search(
        query_path = encoded_query_path,
        index_path = index_path,
        output_path=out_run_path,
        topk=eval_args.topk,
        batch_size = eval_args.threads
    )
    if qrel_path is None:
        return
    metric_scores = pytrec_evaluate(qrel_path, out_run_path)
    for k in metric_scores.keys():
        if k != "perquery":
            logger.info(metric_scores[k])
    json.dump(metric_scores, open(out_metric_path, 'w'), indent=1)
    

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if is_main_process(eval_args.local_rank) else logging.WARN,
    )
    if is_main_process(eval_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    set_seed(2022)

    config = AutoConfig.from_pretrained(model_args.backbone_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.backbone_name_or_path, config=config)
    model = AutoUnicoilModel.from_pretrained(model_args.backbone_name_or_path, config=config)

    if model_args.adapter_name_or_path is not None:
        adapter_name = model.load_adapter(model_args.adapter_name_or_path)
        model.set_active_adapters(adapter_name)
        if model_args.merge_lora:
            model.merge_adapter(adapter_name)
        else:
            logger.info("If your REM has LoRA modules, you can pass --merge_lora argument to merge LoRA weights and speed up inference.")

    out_metric_path = os.path.join(data_args.out_query_dir, "metric.json")
    if os.path.exists(out_metric_path) and not eval_args.overwrite_output_dir:
        logger.info(f"Exit because {out_metric_path} file already exists. ")
        exit(0)

    encoded_query_path = load_or_encode_query(model, tokenizer, data_args.query_path, data_args.out_query_dir, data_args, eval_args)
    torch.cuda.empty_cache()
    index_path = load_or_encode_corpus(model, tokenizer, data_args, eval_args)
    torch.cuda.empty_cache()

    if is_main_process(eval_args.local_rank):
        os.makedirs(data_args.out_query_dir, exist_ok=True)
        search_and_compute_metrics(index_path, encoded_query_path, out_metric_path, data_args.out_query_dir, data_args.qrel_path, eval_args)        


if __name__ == "__main__":
    main()