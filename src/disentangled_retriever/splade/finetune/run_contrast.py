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

from ..modeling import AutoSpladeModel
from .contrast_utils import (
    BackboneContrastSpladeFinetuner,
    AdapterContrastSpladeFinetuner, 
    QDRelDataset, FinetuneCollator
)
from ...adapter_arg import (
    AdapterArguments,
    parse_adapter_arguments
)


logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    qrel_path: str = field()
    query_path: str = field()
    corpus_path: str = field()  
    max_query_len: int = field()
    max_doc_len: int = field()  
    

@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    new_adapter_name: str = field(default=None, metadata={
        "help": "Train a REM module from scratch."})
    init_adapter_path: str = field(default=None, metadata={
        "help": "For example, in few-shot settings, REM module will be further trained in the target domain."})


@dataclass
class SpladeFinetuneArguments(TrainingArguments):
    inv_temperature: float = field(default=1)
    negative: str = field(default="random")
    neg_per_query: int = field(default=1)
    seed: int = field(default=2022)

    q_flops_loss_factor: float = field(default=0.001)
    p_flops_loss_factor: float = field(default=0.001)
    flop_increase_epoch_factor: float = field(default=1)
    flop_log_steps: int = field(default=100)
    
    remove_unused_columns: bool = field(default=False)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((
        AdapterArguments,
        ModelArguments, DataTrainingArguments, SpladeFinetuneArguments))
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
    training_args: SpladeFinetuneArguments

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
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        config = config
    )
    model = AutoSpladeModel.from_pretrained(model_args.model_name_or_path, config=config)

    if model_args.new_adapter_name is None:
        if model_args.init_adapter_path is None:
            logger.info("Add no adapter and only train the backbone")
            trainer_class = BackboneContrastSpladeFinetuner
        else:
            logger.info(f"Init adapter with {model_args.init_adapter_path} and further train it")
            trainer_class = AdapterContrastSpladeFinetuner
            adapter_name = model.load_adapter(model_args.init_adapter_path)
            model.train_adapter(adapter_name)
            for param in model.get_input_embeddings().parameters():
                param.requires_grad = True
            model.active_head = None
            logger.info(f"Parameters with gradient: {[n for n, p in model.named_parameters() if p.requires_grad]}")
    else:
        trainer_class = AdapterContrastSpladeFinetuner
        model_param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
        adapter_config = parse_adapter_arguments(adapter_args)
        model.add_adapter(model_args.new_adapter_name, config=adapter_config)
        model.train_adapter(model_args.new_adapter_name, train_embeddings=True)
        for param in model.get_input_embeddings().parameters():
            param.requires_grad = True
        model.active_head = None # a potential bug in adapter-transformer
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
    eval_dataset = None
    
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
