import os
import sys
import torch
import logging
import transformers
from typing import List
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    AutoAdapterModel,
    set_seed, )
from transformers.trainer_utils import is_main_process
from dataclasses import dataclass, field

from .train_utils import (
    TrainRerankDataset, RerankFinetuneCollator,
    BackboneRerankFinetuner,
    AdapterRerankFinetuner,
    load_validation_set
)
from ...adapter_arg import (
    AdapterArguments,
    parse_adapter_arguments
)

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    qrel_path: str = field()
    query_path: str = field()
    corpus_path: str = field()
    candidate_path: str = field(default=None, metadata={"help": "If NONE, use random negatives"})  
    max_seq_len: int = field(default=512)
    valid_corpus_path: str = field(default=None)
    valid_query_path: str = field(default=None)
    valid_qrel_path: str = field(default=None)
    valid_candidate_path: str = field(default=None)
    valid_topk: int = field(default=None)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: str = field()
    new_adapter_name: str = field(default=None, metadata={
        "help": "Train a REM module from scratch."})
    init_adapter_path: str = field(default=None, metadata={
        "help": "For example, in few-shot settings, REM module will be further trained in the target domain."})


@dataclass
class RerankFinetuneArguments(TrainingArguments):
    seed: int = field(default=2022)
    neg_per_query: int = field(default=3)
    not_train_embedding: bool = field(default=False)

    remove_unused_columns: bool = field(default=False)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((
        AdapterArguments,
        ModelArguments, DataTrainingArguments, RerankFinetuneArguments))
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
    training_args: RerankFinetuneArguments

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
    model = AutoAdapterModel.from_pretrained(model_args.model_name_or_path, config=config)

    if model_args.new_adapter_name is None:
        
        if model_args.init_adapter_path is None:
            logger.info("Add no adapter and only train the backbone")
            trainer_class = BackboneRerankFinetuner
            if hasattr(model, "heads") and "finetune" in list(model.heads):
                logger.info("Initial model already has finetune-head. Will continue use this head.")
            else:
                model.add_classification_head(head_name="finetune", num_labels=1, layers=1, use_pooler=False)
        else:
            logger.info(f"Init adapter with {model_args.init_adapter_path} and further train it")
            trainer_class = AdapterRerankFinetuner
            adapter_name = model.load_adapter(model_args.init_adapter_path)
            model.active_head = adapter_name
            model.train_adapter(adapter_name)

            embedding_path = os.path.join(model_args.init_adapter_path, "embeddings.bin")
            if os.path.exists(embedding_path):
                model.base_model.embeddings.load_state_dict(torch.load(embedding_path, map_location="cpu")) 
                logger.info(f"Overwrite embedding: {embedding_path}")
            else:
                logger.info(f"No new embedding available for overwriting.")
                
            if not training_args.not_train_embedding:
                for n, param in model.base_model.embeddings.named_parameters():
                    param.requires_grad = True
            logger.info(f"Parameters with gradient: {[n for n, p in model.named_parameters() if p.requires_grad]}")
    else:
        trainer_class = AdapterRerankFinetuner 
        model_param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.add_classification_head(head_name=model_args.new_adapter_name, num_labels=1, layers=1, use_pooler=False)
        adapter_config = parse_adapter_arguments(adapter_args)
        model.add_adapter(model_args.new_adapter_name, config=adapter_config)
        model.train_adapter(model_args.new_adapter_name)
        if not training_args.not_train_embedding:
            for n, param in model.base_model.embeddings.named_parameters():
                param.requires_grad = True
        logger.info(f"Parameters with gradient: {[n for n, p in model.named_parameters() if p.requires_grad]}")
        adapter_param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"adapter_param_cnt:{adapter_param_cnt}, model_param_cnt:{model_param_cnt}, ratio:{adapter_param_cnt/model_param_cnt:.4f}")
    
    logger.info(f"Trainer Class: {trainer_class}")
    all_model_param_cnt = sum(p.numel() for p in model.parameters())
    optimize_param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"all_model_param_cnt:{all_model_param_cnt}, optimize_param_cnt:{optimize_param_cnt}, ratio:{optimize_param_cnt/all_model_param_cnt:.4f}")

    train_set = TrainRerankDataset(
            tokenizer = tokenizer, 
            qrel_path = data_args.qrel_path, 
            query_path = data_args.query_path, 
            corpus_path = data_args.corpus_path, 
            candidate_path = data_args.candidate_path,
            neg_per_query = training_args.neg_per_query,
            verbose=is_main_process(training_args.local_rank))
    # Data collator
    data_collator = RerankFinetuneCollator(
        tokenizer = tokenizer,
        max_seq_len = data_args.max_seq_len, 
    )
    if data_args.valid_corpus_path is None:
        eval_dataset = None
    else:
        eval_dataset=load_validation_set(
            data_args.valid_corpus_path,
            data_args.valid_query_path,
            data_args.valid_candidate_path,
            data_args.valid_qrel_path,
        )
        logger.info(f"All validation pairs: {sum((len(v) for v in eval_dataset[2].values()))}")
        if data_args.valid_topk is not None:
            new_candidates = {k: v[:data_args.valid_topk] for k, v in eval_dataset[2].items()}
            eval_dataset = (eval_dataset[0], eval_dataset[1], new_candidates, eval_dataset[3])
            logger.info(f"New validation pairs: {sum((len(v) for v in eval_dataset[2].values()))}")

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_dataset=eval_dataset
    )
    # additionally save checkpoint at the end of one epoch
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if is_main_process(training_args.local_rank):
        trainer.save_model()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
