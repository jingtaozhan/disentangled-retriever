import os
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed, AutoConfig)
from transformers.trainer_utils import is_main_process

from ..modeling import (
    AutoDenseModel, 
    SIMILARITY_METRICS,
    POOLING_METHODS
)
from .contrast_utils import (
    BackboneContrastDenseFinetuner,
    AdapterContrastDenseFinetuner, 
    QDRelDataset, FinetuneCollator
)
from .adapter_arg import (
    AdapterArguments,
    parse_adapter_arguments
)
from .validate_utils import load_validation_set


logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    qrel_path: str = field()
    query_path: str = field()
    corpus_path: str = field()  
    max_query_len: int = field()
    max_doc_len: int = field()  
    valid_corpus_path : str = field(default=None)
    valid_query_path : str = field(default=None)
    valid_qrel_path : str = field(default=None)
    

@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    pooling: str = field(metadata={"choices": POOLING_METHODS})
    similarity_metric: str = field(metadata={"choices": SIMILARITY_METRICS})
    new_adapter_name: str = field(default=None)


@dataclass
class DenseFinetuneArguments(TrainingArguments):
    inv_temperature: float = field(default=1)
    negative: str = field(default="random")
    neg_per_query: int = field(default=1)
    seed: int = field(default=2022)

    remove_unused_columns: bool = field(default=False)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((
        AdapterArguments,
        ModelArguments, DataTrainingArguments, DenseFinetuneArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        adapter_args, model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        adapter_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
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
    training_args: DenseFinetuneArguments

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
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    logger.info("Training parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.similarity_metric = model_args.similarity_metric
    config.pooling = model_args.pooling    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        config = config
    )
    model = AutoDenseModel.from_pretrained(model_args.model_name_or_path, config=config)

    if model_args.new_adapter_name is None:
        logger.info("Add no adapter and only train the backbone")
        trainer_class = BackboneContrastDenseFinetuner
    else:
        trainer_class = AdapterContrastDenseFinetuner
        model_param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
        adapter_config = parse_adapter_arguments(adapter_args)
        model.add_adapter(model_args.new_adapter_name, config=adapter_config)
        model.train_adapter(model_args.new_adapter_name)
        logger.info(f"Parameters with gradient: {[n for n, p in model.named_parameters() if p.requires_grad]}")
        adapter_param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"adapter_param_cnt:{adapter_param_cnt}, model_param_cnt:{model_param_cnt}, ratio:{adapter_param_cnt/model_param_cnt:.4f}")

    logger.info(f"Trainer Class: {trainer_class}")
    all_model_param_cnt = sum(p.numel() for p in model.parameters())
    optimize_param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"all_model_param_cnt:{all_model_param_cnt}, optimize_param_cnt:{optimize_param_cnt}, ratio:{optimize_param_cnt/all_model_param_cnt:.4f}")

    train_set = QDRelDataset(tokenizer, 
            qrel_path = data_args.qrel_path, 
            query_path = data_args.query_path, 
            corpus_path = data_args.corpus_path, 
            max_query_len = data_args.max_query_len, 
            max_doc_len = data_args.max_doc_len, 
            rel_threshold = 1, 
            negative = training_args.negative,
            neg_per_query = training_args.neg_per_query,
            verbose=is_main_process(training_args.local_rank))
    # Data collator
    data_collator = FinetuneCollator(
        tokenizer = tokenizer,
        max_query_len = data_args.max_query_len, 
        max_doc_len = data_args.max_doc_len,
    )
    if data_args.valid_corpus_path is None:
        eval_dataset = None
        assert data_args.valid_query_path is None and data_args.valid_qrel_path is None
    else:
        eval_dataset=load_validation_set(
            data_args.valid_corpus_path,
            data_args.valid_query_path,
            data_args.valid_qrel_path,
        )
    
    # Initialize our Trainer
    trainer = trainer_class(
        qrels=train_set.get_qrels(),
        model=model,
        args=training_args,
        train_dataset=train_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_dataset=eval_dataset
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
