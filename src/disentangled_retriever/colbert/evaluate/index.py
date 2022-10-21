import os
import time
import torch
import ujson
import numpy as np
import logging
import itertools
import threading
import queue
from dataclasses import field, dataclass
import transformers
from transformers import (
    TrainingArguments,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser
)
from transformers.trainer_utils import is_main_process

from ..modeling import AutoColBERTModel

from .model_inference import ModelInference

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    corpus_path: str = field()    
    max_doc_len: int = field(default=512)  


@dataclass
class ModelArguments:
    backbone_name_or_path: str = field()
    adapter_name_or_path: str = field(default=None)
    merge_lora: bool = field(default=False)


@dataclass
class EvalArguments(TrainingArguments):
    topk : int = field(default=1000)
    chunksize: float = field(default=1.0)  # in GiBs
    output_dim: int = field(default=32)


class IndexManager():
    def save(self, tensor, path_prefix):
        torch.save(tensor, path_prefix)


class CollectionEncoder():
    def __init__(self, data_args: DataArguments, model_args: ModelArguments, eval_args: EvalArguments):
        self.data_args = data_args
        self.model_args = model_args
        self.eval_args = eval_args
        self.process_idx = eval_args.local_rank
        self.num_processes = eval_args.world_size

        assert 0.5 <= eval_args.chunksize <= 128.0
        max_bytes_per_file = eval_args.chunksize * (1024*1024*1024)

        max_bytes_per_doc = (data_args.max_doc_len * eval_args.output_dim * 2.0)

        # Determine subset sizes for output
        minimum_subset_size = 10_000
        maximum_subset_size = max_bytes_per_file / max_bytes_per_doc
        maximum_subset_size = max(minimum_subset_size, maximum_subset_size)
        self.possible_subset_sizes = [int(maximum_subset_size)]

        self.print_main("#> Local args.bsize =", eval_args.per_gpu_eval_batch_size)
        self.print_main("#> args.index_root =", eval_args.output_dir)
        self.print_main(f"#> self.possible_subset_sizes = {self.possible_subset_sizes}")

        self._load_model()
        self.indexmgr = IndexManager()
        self.iterator = self._initialize_iterator()

    def _initialize_iterator(self):
        return open(self.data_args.corpus_path)

    def _saver_thread(self):
        for args in iter(self.saver_queue.get, None):
            self._save_batch(*args)

    def _load_model(self):
        config = AutoConfig.from_pretrained(self.model_args.backbone_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_args.backbone_name_or_path, config=config)
        model = AutoColBERTModel.from_pretrained(self.model_args.backbone_name_or_path, config=config)

        if self.model_args.adapter_name_or_path is not None:
            adapter_name = model.load_adapter(self.model_args.adapter_name_or_path)
            model.set_active_adapters(adapter_name)
            if self.model_args.merge_lora:
                model.merge_adapter(adapter_name)
            else:
                print("If your REM has LoRA modules, you can pass --merge_lora argument to merge LoRA weights and speed up inference.")
        
        self.colbert = model
        self.colbert = self.colbert.to(self.eval_args.device)
        self.colbert.eval()

        self.inference = ModelInference(self.colbert, tokenizer=tokenizer, max_query_len=None, max_doc_len=self.data_args.max_doc_len)

    def encode(self):
        self.saver_queue = queue.Queue(maxsize=3)
        thread = threading.Thread(target=self._saver_thread)
        thread.start()

        t0 = time.time()
        local_docs_processed = 0

        self.print("Ignore pids, use line offsets as pids")

        for batch_idx, (offset, lines, owner) in enumerate(self._batch_passages(self.iterator)):
            if owner != self.process_idx:
                continue

            t1 = time.time()
            batch = self._preprocess_batch(offset, lines)
            embs, doclens = self._encode_batch(batch_idx, batch)

            t2 = time.time()
            self.saver_queue.put((batch_idx, embs, offset, doclens))

            t3 = time.time()
            local_docs_processed += len(lines)
            overall_throughput = compute_throughput(local_docs_processed, t0, t3)
            this_encoding_throughput = compute_throughput(len(lines), t1, t2)
            this_saving_throughput = compute_throughput(len(lines), t2, t3)

            self.print(f'#> Completed batch #{batch_idx} (starting at passage #{offset}) \t\t'
                          f'Passages/min: {overall_throughput} (overall), ',
                          f'{this_encoding_throughput} (this encoding), ',
                          f'{this_saving_throughput} (this saving)')
        self.saver_queue.put(None)

        self.print("#> Joining saver thread.")
        thread.join()

    def _batch_passages(self, fi):
        """
        Must use the same seed across processes!
        """
        np.random.seed(0)

        offset = 0
        for owner in itertools.cycle(range(self.num_processes)):
            batch_size = np.random.choice(self.possible_subset_sizes)

            L = [line for _, line in zip(range(batch_size), fi)]

            if len(L) == 0:
                break  # EOF

            yield (offset, L, owner)
            offset += len(L)

            if len(L) < batch_size:
                break  # EOF

        self.print("[NOTE] Done with local share.")

        return

    def _preprocess_batch(self, offset, lines):

        endpos = offset + len(lines)

        batch = []

        for line_idx, line in zip(range(offset, endpos), lines):
            line_parts = line.split('\t')

            pid, passage = line_parts

            assert len(passage) >= 1

            batch.append(passage)

            # assert pid == 'id' or int(pid) == line_idx

        return batch

    def _encode_batch(self, batch_idx, batch):
        with torch.no_grad():
            embs = self.inference.docFromText(batch, bsize=self.eval_args.per_device_eval_batch_size, keep_dims=False)
            assert type(embs) is list
            assert len(embs) == len(batch)

            local_doclens = [d.size(0) for d in embs]
            embs = torch.cat(embs)

        return embs, local_doclens

    def _save_batch(self, batch_idx, embs, offset, doclens):
        start_time = time.time()

        output_path = os.path.join(self.eval_args.output_dir, "{}.pt".format(batch_idx))
        output_sample_path = os.path.join(self.eval_args.output_dir, "{}.sample".format(batch_idx))
        doclens_path = os.path.join(self.eval_args.output_dir, 'doclens.{}.json'.format(batch_idx))

        # Save the embeddings.
        self.indexmgr.save(embs, output_path)
        self.indexmgr.save(embs[torch.randint(0, high=embs.size(0), size=(embs.size(0) // 20,))], output_sample_path)

        # Save the doclens.
        with open(doclens_path, 'w') as output_doclens:
            ujson.dump(doclens, output_doclens)

        throughput = compute_throughput(len(doclens), start_time, time.time())
        self.print_main("#> Saved batch #{} to {} \t\t".format(batch_idx, output_path),
                        "Saving Throughput =", throughput, "passages per minute.\n")

    def print(self, *args):
        print("[" + str(self.process_idx) + "]", "\t\t", *args)

    def print_main(self, *args):
        if self.process_idx == 0:
            self.print(*args)


def compute_throughput(size, t0, t1):
    throughput = size / (t1 - t0) * 60

    if throughput > 1000 * 1000:
        throughput = throughput / (1000*1000)
        throughput = round(throughput, 1)
        return '{}M'.format(throughput)

    throughput = throughput / (1000)
    throughput = round(throughput, 1)
    return '{}k'.format(throughput)


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

        os.makedirs(eval_args.output_dir, exist_ok=True)

    CollectionEncoder(data_args, model_args, eval_args).encode() 

    if is_main_process(eval_args.local_rank):
        docids = [line.split("\t")[0] for line in open(data_args.corpus_path)]
        ujson.dump(docids, open(os.path.join(eval_args.output_dir, "docids.json"), 'w'))


if __name__ == "__main__":
    main()