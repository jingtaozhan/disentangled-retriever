import os
import re
import sys
import gzip
import math
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Optional, List, Dict
from dataclasses import dataclass, field

import transformers
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForPreTraining,
    Trainer,
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    PreTrainedTokenizer,
)
from transformers.trainer_utils import is_main_process
from transformers.data import DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    corpus_path: str = field()
    max_seq_length: int = field()
    min_sentence_length: int = field()
    sentence_tokenization: str = field(metadata={"choices":["chinese_re"]})


@dataclass
class ModelArguments:
    model_name_or_path: str = field()


class TextDatasetForNextSentencePrediction(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        corpus_sentences: List[List[str]],
        block_size: int,
        short_seq_probability=0.1,
        nsp_probability=0.5,
        verbose=True
    ):
        self.short_seq_probability = short_seq_probability
        self.nsp_probability = nsp_probability

        self.tokenizer = tokenizer
        self.documents = []
        for sentences in tqdm(corpus_sentences, disable=not verbose, desc="Tokenizing"):
            self.documents.append([])
            for sent in sentences:
                tokens = tokenizer.tokenize(sent)
                tokens = tokenizer.convert_tokens_to_ids(tokens)
                self.documents[-1].append(tokens)

        logger.info(f"Creating examples from {len(self.documents)} documents.")
        self.examples = []
        for doc_index, document in enumerate(tqdm(self.documents, disable=not verbose, desc="Constructing Pairs")):
            self.create_examples_from_document(document, doc_index, block_size)

    def create_examples_from_document(self, document: List[List[int]], doc_index: int, block_size: int):
        """Creates examples for a single document."""

        max_num_tokens = block_size - self.tokenizer.num_special_tokens_to_add(pair=True)

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pretraining and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_probability:
            target_seq_length = random.randint(2, max_num_tokens)

        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0

        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []

                    if len(current_chunk) == 1 or random.random() < self.nsp_probability:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(10):
                            random_document_index = random.randint(0, len(self.documents) - 1)
                            if random_document_index != doc_index:
                                break

                        random_document = self.documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    # Actual next
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    if not (len(tokens_a) >= 1):
                        raise ValueError(f"Length of sequence a is {len(tokens_a)} which must be no less than 1")
                    if not (len(tokens_b) >= 1):
                        raise ValueError(f"Length of sequence b is {len(tokens_b)} which must be no less than 1")

                    # Truncation
                    if len(tokens_a) + len(tokens_b) > max_num_tokens:
                        shorter_length = min(len(tokens_a), len(tokens_b))
                        if 2 * shorter_length > max_num_tokens:
                            # truncate both sentences to the same length
                            tokens_a = tokens_a[:math.floor(max_num_tokens/2)]
                            tokens_b = tokens_b[:math.ceil(max_num_tokens/2)]
                        else:
                            # only truncate the longer one 
                            longer_length = max_num_tokens - shorter_length
                            tokens_a = tokens_a[:longer_length]
                            tokens_b = tokens_b[:longer_length]

                    # add special tokens
                    input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                        "next_sentence_label": torch.tensor(1 if is_random_next else 0, dtype=torch.long),
                    }

                    self.examples.append(example)

                current_chunk = []
                current_length = 0

            i += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def read_corpus(file_path, verbose):
    logger.info(f"Load corpus: {file_path}")
    open_function = gzip.open if file_path.endswith(".gz") else open
    dataset = []
    for idx, line in tqdm(enumerate(open_function(file_path)), disable=not verbose, mininterval=10):
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        splits = line.split("\t")
        if len(splits) == 2:
            _id, text = splits
        else:
            raise NotImplementedError("Corpus Format: id\\ttext\\n")
        dataset.append(text)
    return dataset


def chinese_re_tokenize_sentence(paragraph, min_length=None):
    sentences = list(re.findall(u'[^!?。\.\!\?]+[!?。\.\!\?]?', paragraph, flags=re.U))
    if min_length is not None:
        sentences = list(filter(lambda x: len(x) > min_length, sentences))
    return sentences

        
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    
    resume_from_checkpoint = False
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and any((x.startswith("checkpoint") for x in os.listdir(training_args.output_dir)))
    ):
        if not training_args.overwrite_output_dir:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        else:
            resume_from_checkpoint = True

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
    model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
    
    corpus = read_corpus(data_args.corpus_path, verbose=is_main_process(training_args.local_rank))
    if data_args.sentence_tokenization == "chinese_re":
        tokenized_corpus = [chinese_re_tokenize_sentence(psg, data_args.min_sentence_length) 
            for psg in tqdm(corpus, disable=not is_main_process(training_args.local_rank))]
    else:
        raise NotImplementedError()
    train_set = TextDatasetForNextSentencePrediction(
        tokenizer,
        list(filter(lambda x: len(x)>=2, tokenized_corpus)),
        data_args.max_seq_length,
        verbose=is_main_process(training_args.local_rank)
    )
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if is_main_process(training_args.local_rank):
        trainer.save_model()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
