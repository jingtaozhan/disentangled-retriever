import logging
from dataclasses import dataclass, field
from transformers.adapters import (
    AdapterConfig, 
    LoRAConfig, 
    ParallelConfig,
    PrefixTuningConfig, 
    MAMConfig,
    ConfigUnion
)

logger = logging.getLogger(__name__)


@dataclass
class AdapterArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    bottleneck_reduction_factor: float = field(default=None, metadata={"help": "If not None, add a bottleneck adapter and only trains this adapter."})

    parallel_reduction_factor: float = field(default=None, metadata={"help": "If not None, add a parallel adapter and only trains this adapter."})
    parallel_scaling: float = field(default = 4.0)

    lora_rank: int = field(default=None, metadata={"help": "If not None, add a lora adapter and only trains this adapter."})
    lora_alpha: int = field(default=None)

    mam_parallel_reduction_factor: float = field(default=None, metadata={"help": "If not None, add a mam-parallel adapter and only trains this adapter."})
    mam_prefix_length: int = field(default=None, metadata={"help": "If not None, add a mam-prefix adapter and only trains this adapter."})


def parse_adapter_arguments(
    adapter_args: AdapterArguments,
):
    adapter_config_lst = []
    if adapter_args.bottleneck_reduction_factor is not None:
        logger.info("Add a bottleneck adapter and only train the adapter")
        adapter_config = AdapterConfig(
            mh_adapter = True, 
            output_adapter = True, 
            reduction_factor = adapter_args.bottleneck_reduction_factor, 
            non_linearity = "relu"
        )
        adapter_config_lst.append(adapter_config)
    if adapter_args.lora_rank is not None:
        logger.info("Add a lora adapter and only train the adapter")
        adapter_config = LoRAConfig(
            r = adapter_args.lora_rank, 
            alpha = adapter_args.lora_alpha if adapter_args.lora_alpha is not None else adapter_args.lora_rank
        )
        adapter_config_lst.append(adapter_config)
    if adapter_args.parallel_reduction_factor is not None:
        logger.info("Add a parallel adapter and only train the adapter")
        adapter_config = ParallelConfig(
            reduction_factor=adapter_args.parallel_reduction_factor,
            scaling=adapter_args.parallel_scaling
        )
        adapter_config_lst.append(adapter_config)
    if adapter_args.mam_parallel_reduction_factor is not None:
        logger.info("Add a MAM adapter and only train the adapter")
        adapter_config = MAMConfig(
            PrefixTuningConfig(prefix_length=adapter_args.mam_prefix_length, bottleneck_size=880), 
            ParallelConfig(reduction_factor=adapter_args.mam_parallel_reduction_factor)
        )
        adapter_config_lst.append(adapter_config)
    
    if len(adapter_config_lst) == 1:
        return adapter_config_lst[0]
    elif len(adapter_config_lst) > 1:
        return ConfigUnion(*adapter_config_lst)
    else:
        raise RuntimeError(f"{adapter_config_lst} is empty")