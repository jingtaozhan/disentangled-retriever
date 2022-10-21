from collections import defaultdict
import os
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from dataclasses import field, dataclass

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    HfArgumentParser, 
    TrainingArguments,
    AutoAdapterModel,
    set_seed)
from transformers.trainer_utils import is_main_process


from .eval_utils import (
    rerank,
    load_corpus, 
    load_queries,
    load_candidates,
)
from ...evaluate import pytrec_evaluate

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    corpus_path: str = field()
    query_path: str = field()
    candidate_path: str = field()
    qrel_path: str = field(default=None)
    

@dataclass
class ModelArguments:
    backbone_name_or_path: str = field()
    adapter_name_or_path: str = field(default=None)
    merge_lora: bool = field(default=False)


@dataclass
class EvalArguments(TrainingArguments):
    topk : int = field(default=1000)


def output_and_compute_metrics(score_dict, out_metric_path, out_run_path, qrel_path):
    with open(out_run_path, 'w') as output:
        for qid, pid_list in score_dict.items():
            for i, (docid, score) in enumerate(pid_list):
                output.write(f"{qid}\tQ0\t{docid}\t{i+1}\t{score}\tSystem\n")

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
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(eval_args.local_rank) else logging.WARN,
    )
    if is_main_process(eval_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    set_seed(2022)

    config = AutoConfig.from_pretrained(model_args.backbone_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.backbone_name_or_path, config=config)
    model = AutoAdapterModel.from_pretrained(model_args.backbone_name_or_path, config=config)

    if model_args.adapter_name_or_path is not None:
        adapter_name = model.load_adapter(model_args.adapter_name_or_path)
        model.set_active_adapters(adapter_name)
        embedding_path = os.path.join(model_args.adapter_name_or_path, "embeddings.bin")
        if os.path.exists(embedding_path):
            model.base_model.embeddings.load_state_dict(torch.load(embedding_path, map_location="cpu")) 
            logger.info(f"Overwrite embedding: {embedding_path}")
        else:
            logger.info(f"No new embedding available for overwriting.")
        if model_args.merge_lora:
            model.merge_adapter(adapter_name)
        else:
            logger.info("If your REM has LoRA modules, you can pass --merge_lora argument to merge LoRA weights and speed up inference.")

    out_metric_path = os.path.join(eval_args.output_dir, f"rerank{eval_args.topk}.metric.json")
    if os.path.exists(out_metric_path) and not eval_args.overwrite_output_dir:
        logger.info(f"Exit because {out_metric_path} file already exists. ")
        exit(0)

    corpus = load_corpus(data_args.corpus_path, verbose=is_main_process(eval_args.local_rank))
    queries = load_queries(data_args.query_path)
    candidates = load_candidates(data_args.candidate_path, verbose=is_main_process(eval_args.local_rank), topk=eval_args.topk)

    score_dict = rerank(
        queries = queries,
        corpus = corpus,
        candidates = candidates,
        model = model, 
        tokenizer = tokenizer, 
        eval_args = eval_args
    )

    if is_main_process(eval_args.local_rank):
        out_run_path = os.path.join(eval_args.output_dir, f"rerank{eval_args.topk}.run.tsv")
        output_and_compute_metrics(score_dict, out_metric_path, out_run_path, data_args.qrel_path)        


if __name__ == "__main__":
    main()