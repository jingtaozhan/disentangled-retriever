import re
import os
import sys
import json
import math
import torch
import faiss
import logging
import subprocess
import numpy as np
from tqdm import tqdm
import faiss.contrib.torch_utils
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import TrainingArguments, Trainer

from ...dense.evaluate.index_utils import load_corpus, load_queries, TextDataset, get_collator_func
from ...splade.evaluate.index_utils import SpladeEvaluater, encode_sparse_text, batch_anserini_search, create_anserini_index
logger = logging.getLogger(__name__)


class UnicoilEvaluater(SpladeEvaluater):
    rounding_func = torch.ceil


def encode_unicoil_text(
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
        UnicoilEvaluater,
        corpus, 
        model, 
        tokenizer, 
        eval_args, 
        encode_is_qry,
        encoded_save_path,
        split_corpus_num=split_corpus_num, 
        verbose=verbose
    )