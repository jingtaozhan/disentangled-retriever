import os
import time
import ujson
import torch
import faiss
import random
import logging
import itertools
from functools import partial
from itertools import accumulate
from multiprocessing import Pool
from contextlib import contextmanager
from dataclasses import field, dataclass

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    HfArgumentParser, 
    TrainingArguments,
    set_seed)
from transformers.trainer_utils import is_main_process

from .index_faiss import print_message, load_doclens
from .model_inference import ModelInference
from ..modeling import AutoColBERTModel
from disentangled_retriever.evaluate import pytrec_evaluate


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    index_path: str = field()
    query_path: str = field()
    max_query_len: int = field(default=32)
    qrel_path: str = field(default=None)

    def __post_init__(self):
        self.faiss_index_path = os.path.join(self.index_path, "ivfpq.faiss")
        self.docids_path = os.path.join(self.index_path, "docids.json")

@dataclass
class ModelArguments:
    backbone_name_or_path: str = field()
    adapter_name_or_path: str = field(default=None)
    merge_lora: bool = field(default=False)
    output_dim: int = field(default=32)


@dataclass
class EvalArguments(TrainingArguments):
    faiss_depth: int = field(default=1024)
    part_range: int = field(default=None)
    depth: int = field(default=1000)
    nprobe: int = field(default=10)

    def __post_init__(self):
        super().__post_init__()
        if self.part_range:
            part_offset, part_endpos = map(int, self.part_range.split('..'))
            self.part_range = range(part_offset, part_endpos)
        self.output_ranking_path = os.path.join(self.output_dir, "run.tsv")
        self.output_metric_path = os.path.join(self.output_dir, "metric.json")


def load_index_part(filename, verbose=True):
    part = torch.load(filename)

    if type(part) == list:  # for backward compatibility
        part = torch.cat(part)

    return part

def get_parts(directory):
    extension = '.pt'

    parts = sorted([int(filename[: -1 * len(extension)]) for filename in os.listdir(directory)
                    if filename.endswith(extension)])

    assert list(range(len(parts))) == parts, parts

    # Integer-sortedness matters.
    parts_paths = [os.path.join(directory, '{}{}'.format(filename, extension)) for filename in parts]
    samples_paths = [os.path.join(directory, '{}.sample'.format(filename)) for filename in parts]

    return parts, parts_paths, samples_paths


def flatten(L):
    return [x for y in L for x in y]


class FaissIndex():
    def __init__(self, index_path, faiss_index_path, nprobe, part_range=None):
        print_message("#> Loading the FAISS index from", faiss_index_path, "..")

        faiss_part_range = os.path.basename(faiss_index_path).split('.')[-2].split('-')

        if len(faiss_part_range) == 2:
            faiss_part_range = range(*map(int, faiss_part_range))
            assert part_range[0] in faiss_part_range, (part_range, faiss_part_range)
            assert part_range[-1] in faiss_part_range, (part_range, faiss_part_range)
        else:
            faiss_part_range = None

        self.part_range = part_range
        self.faiss_part_range = faiss_part_range

        self.faiss_index = faiss.read_index(faiss_index_path)
        self.faiss_index.nprobe = nprobe

        print_message("#> Building the emb2pid mapping..")
        all_doclens = load_doclens(index_path, flatten=False)

        pid_offset = 0
        if faiss_part_range is not None:
            print(f"#> Restricting all_doclens to the range {faiss_part_range}.")
            pid_offset = len(flatten(all_doclens[:faiss_part_range.start]))
            all_doclens = all_doclens[faiss_part_range.start:faiss_part_range.stop]

        self.relative_range = None
        if self.part_range is not None:
            start = self.faiss_part_range.start if self.faiss_part_range is not None else 0
            a = len(flatten(all_doclens[:self.part_range.start - start]))
            b = len(flatten(all_doclens[:self.part_range.stop - start]))
            self.relative_range = range(a, b)
            print(f"self.relative_range = {self.relative_range}")

        all_doclens = flatten(all_doclens)

        total_num_embeddings = sum(all_doclens)
        self.emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

        offset_doclens = 0
        for pid, dlength in enumerate(all_doclens):
            self.emb2pid[offset_doclens: offset_doclens + dlength] = pid_offset + pid
            offset_doclens += dlength

        print_message("len(self.emb2pid) =", len(self.emb2pid))

        self.parallel_pool = Pool(16)

    def retrieve(self, faiss_depth, Q, verbose=False):
        embedding_ids = self.queries_to_embedding_ids(faiss_depth, Q, verbose=verbose)
        pids = self.embedding_ids_to_pids(embedding_ids, verbose=verbose)

        if self.relative_range is not None:
            pids = [[pid for pid in pids_ if pid in self.relative_range] for pids_ in pids]

        return pids

    def queries_to_embedding_ids(self, faiss_depth, Q, verbose=True):
        # Flatten into a matrix for the faiss search.
        num_queries, embeddings_per_query, dim = Q.size()
        Q_faiss = Q.view(num_queries * embeddings_per_query, dim).cpu().contiguous()

        # Search in large batches with faiss.
        print_message("#> Search in batches with faiss. \t\t",
                      f"Q.size() = {Q.size()}, Q_faiss.size() = {Q_faiss.size()}",
                      condition=verbose)

        embeddings_ids = []
        faiss_bsize = embeddings_per_query * 5000
        for offset in range(0, Q_faiss.size(0), faiss_bsize):
            endpos = min(offset + faiss_bsize, Q_faiss.size(0))

            print_message("#> Searching from {} to {}...".format(offset, endpos), condition=verbose)

            some_Q_faiss = Q_faiss[offset:endpos].float().numpy()
            _, some_embedding_ids = self.faiss_index.search(some_Q_faiss, faiss_depth)
            embeddings_ids.append(torch.from_numpy(some_embedding_ids))

        embedding_ids = torch.cat(embeddings_ids)

        # Reshape to (number of queries, non-unique embedding IDs per query)
        embedding_ids = embedding_ids.view(num_queries, embeddings_per_query * embedding_ids.size(1))

        return embedding_ids

    def embedding_ids_to_pids(self, embedding_ids, verbose=True):
        # Find unique PIDs per query.
        print_message("#> Lookup the PIDs..", condition=verbose)
        all_pids = self.emb2pid[embedding_ids]

        print_message(f"#> Converting to a list [shape = {all_pids.size()}]..", condition=verbose)
        all_pids = all_pids.tolist()

        print_message("#> Removing duplicates (in parallel if large enough)..", condition=verbose)

        if len(all_pids) > 5000:
            all_pids = list(self.parallel_pool.map(uniq, all_pids))
        else:
            all_pids = list(map(uniq, all_pids))

        print_message("#> Done with embedding_ids_to_pids().", condition=verbose)

        return all_pids


def uniq(l):
    return list(set(l))


class IndexRanker():
    def __init__(self, tensor, doclens, device):
        self.tensor = tensor
        self.doclens = doclens

        self.maxsim_dtype = torch.float32
        self.doclens_pfxsum = [0] + list(accumulate(self.doclens))

        self.doclens = torch.tensor(self.doclens)
        self.doclens_pfxsum = torch.tensor(self.doclens_pfxsum)

        self.dim = self.tensor.size(-1)

        self.strides = [torch_percentile(self.doclens, p) for p in [90]]
        self.strides.append(self.doclens.max().item())
        self.strides = sorted(list(set(self.strides)))

        print_message(f"#> Using strides {self.strides}..")

        self.views = self._create_views(self.tensor)
        self.bsize = 1 << 14
        self.buffers = self._create_buffers(self.bsize, self.tensor.dtype, {'cpu', 'cuda:0'})
        self.device = device

    def _create_views(self, tensor):
        views = []

        for stride in self.strides:
            outdim = tensor.size(0) - stride + 1
            view = torch.as_strided(tensor, (outdim, stride, self.dim), (self.dim, self.dim, 1))
            views.append(view)

        return views

    def _create_buffers(self, max_bsize, dtype, devices):
        buffers = {}

        for device in devices:
            buffers[device] = [torch.zeros(max_bsize, stride, self.dim, dtype=dtype,
                                           device=device, pin_memory=(device == 'cpu'))
                               for stride in self.strides]

        return buffers

    def rank(self, Q, pids, views=None, shift=0):
        assert len(pids) > 0
        assert Q.size(0) in [1, len(pids)]

        Q = Q.contiguous().to(self.device).to(dtype=self.maxsim_dtype)

        views = self.views if views is None else views
        VIEWS_DEVICE = views[0].device

        D_buffers = self.buffers[str(VIEWS_DEVICE)]

        raw_pids = pids if type(pids) is list else pids.tolist()
        pids = torch.tensor(pids) if type(pids) is list else pids

        doclens, offsets = self.doclens[pids], self.doclens_pfxsum[pids]

        assignments = (doclens.unsqueeze(1) > torch.tensor(self.strides).unsqueeze(0) + 1e-6).sum(-1)

        one_to_n = torch.arange(len(raw_pids))
        output_pids, output_scores, output_permutation = [], [], []

        for group_idx, stride in enumerate(self.strides):
            locator = (assignments == group_idx)

            if locator.sum() < 1e-5:
                continue

            group_pids, group_doclens, group_offsets = pids[locator], doclens[locator], offsets[locator]
            group_Q = Q if Q.size(0) == 1 else Q[locator]

            group_offsets = group_offsets.to(VIEWS_DEVICE) - shift
            group_offsets_uniq, group_offsets_expand = torch.unique_consecutive(group_offsets, return_inverse=True)

            D_size = group_offsets_uniq.size(0)
            D = torch.index_select(views[group_idx], 0, group_offsets_uniq, out=D_buffers[group_idx][:D_size])
            D = D.to(self.device)
            D = D[group_offsets_expand.to(self.device)].to(dtype=self.maxsim_dtype)

            mask = torch.arange(stride, device=self.device) + 1
            mask = mask.unsqueeze(0) <= group_doclens.to(self.device).unsqueeze(-1)

            scores = (D @ group_Q) * mask.unsqueeze(-1)
            scores = scores.max(1).values.sum(-1).cpu()

            output_pids.append(group_pids)
            output_scores.append(scores)
            output_permutation.append(one_to_n[locator])

        output_permutation = torch.cat(output_permutation).sort().indices
        output_pids = torch.cat(output_pids)[output_permutation].tolist()
        output_scores = torch.cat(output_scores)[output_permutation].tolist()

        assert len(raw_pids) == len(output_pids)
        assert len(raw_pids) == len(output_scores)
        assert raw_pids == output_pids

        return output_scores

    def batch_rank(self, all_query_embeddings, all_query_indexes, all_pids, sorted_pids):
        assert sorted_pids is True

        ######

        scores = []
        range_start, range_end = 0, 0

        for pid_offset in range(0, len(self.doclens), 50_000):
            pid_endpos = min(pid_offset + 50_000, len(self.doclens))

            range_start = range_start + (all_pids[range_start:] < pid_offset).sum()
            range_end = range_end + (all_pids[range_end:] < pid_endpos).sum()

            pids = all_pids[range_start:range_end]
            query_indexes = all_query_indexes[range_start:range_end]

            print_message(f"###--> Got {len(pids)} query--passage pairs in this sub-range {(pid_offset, pid_endpos)}.")

            if len(pids) == 0:
                continue

            print_message(f"###--> Ranking in batches the pairs #{range_start} through #{range_end} in this sub-range.")

            tensor_offset = self.doclens_pfxsum[pid_offset].item()
            tensor_endpos = self.doclens_pfxsum[pid_endpos].item() + 512

            collection = self.tensor[tensor_offset:tensor_endpos].to(self.device)
            views = self._create_views(collection)

            print_message(f"#> Ranking in batches of {self.bsize} query--passage pairs...")

            for batch_idx, offset in enumerate(range(0, len(pids), self.bsize)):
                if batch_idx % 100 == 0:
                    print_message("#> Processing batch #{}..".format(batch_idx))

                endpos = offset + self.bsize
                batch_query_index, batch_pids = query_indexes[offset:endpos], pids[offset:endpos]

                Q = all_query_embeddings[batch_query_index]

                scores.extend(self.rank(Q, batch_pids, views, shift=tensor_offset))

        return scores


def torch_percentile(tensor, p):
    assert p in range(1, 100+1)
    assert tensor.dim() == 1

    return tensor.kthvalue(int(p * tensor.size(0) / 100.0)).values.item()


class IndexPart():
    def __init__(self, directory, device, dim=128, part_range=None, verbose=True):
        first_part, last_part = (0, None) if part_range is None else (part_range.start, part_range.stop)

        # Load parts metadata
        all_parts, all_parts_paths, _ = get_parts(directory)
        self.parts = all_parts[first_part:last_part]
        self.parts_paths = all_parts_paths[first_part:last_part]

        # Load doclens metadata
        all_doclens = load_doclens(directory, flatten=False)

        self.doc_offset = sum([len(part_doclens) for part_doclens in all_doclens[:first_part]])
        self.doc_endpos = sum([len(part_doclens) for part_doclens in all_doclens[:last_part]])
        self.pids_range = range(self.doc_offset, self.doc_endpos)

        self.parts_doclens = all_doclens[first_part:last_part]
        self.doclens = flatten(self.parts_doclens)
        self.num_embeddings = sum(self.doclens)

        self.tensor = self._load_parts(dim, verbose)
        self.ranker = IndexRanker(self.tensor, self.doclens, device)

    def _load_parts(self, dim, verbose):
        tensor = torch.zeros(self.num_embeddings + 512, dim, dtype=torch.float16)

        if verbose:
            print_message("tensor.size() = ", tensor.size())

        offset = 0
        for idx, filename in enumerate(self.parts_paths):
            print_message("|> Loading", filename, "...", condition=verbose)

            endpos = offset + sum(self.parts_doclens[idx])
            part = load_index_part(filename, verbose=verbose)

            tensor[offset:endpos] = part
            offset = endpos

        return tensor

    def pid_in_range(self, pid):
        return pid in self.pids_range

    def rank(self, Q, pids):
        """
        Rank a single batch of Q x pids (e.g., 1k--10k pairs).
        """

        assert Q.size(0) in [1, len(pids)], (Q.size(0), len(pids))
        assert all(pid in self.pids_range for pid in pids), self.pids_range

        pids_ = [pid - self.doc_offset for pid in pids]
        scores = self.ranker.rank(Q, pids_)

        return scores

    def batch_rank(self, all_query_embeddings, query_indexes, pids, sorted_pids):
        """
        Rank a large, fairly dense set of query--passage pairs (e.g., 1M+ pairs).
        Higher overhead, much faster for large batches.
        """

        assert ((pids >= self.pids_range.start) & (pids < self.pids_range.stop)).sum() == pids.size(0)

        pids_ = pids - self.doc_offset
        scores = self.ranker.batch_rank(all_query_embeddings, query_indexes, pids_, sorted_pids)

        return scores


class Ranker():
    def __init__(self, index_path, faiss_index_path, nprobe, part_range, dim, inference, device, faiss_depth=1024):
        self.inference = inference
        self.faiss_depth = faiss_depth

        if faiss_depth is not None:
            self.faiss_index = FaissIndex(index_path, faiss_index_path, nprobe, part_range=part_range)
            self.retrieve = partial(self.faiss_index.retrieve, self.faiss_depth)

        self.index = IndexPart(index_path, device=device, dim=dim, part_range=part_range, verbose=True)

    def encode(self, queries):
        assert type(queries) in [list, tuple], type(queries)

        Q = self.inference.queryFromText(queries, bsize=512 if len(queries) > 512 else None)

        return Q

    def rank(self, Q, pids=None):
        pids = self.retrieve(Q, verbose=False)[0] if pids is None else pids

        assert type(pids) in [list, tuple], type(pids)
        assert Q.size(0) == 1, (len(pids), Q.size())
        assert all(type(pid) is int for pid in pids)

        scores = []
        if len(pids) > 0:
            Q = Q.permute(0, 2, 1)
            scores = self.index.rank(Q, pids)

            scores_sorter = torch.tensor(scores).sort(descending=True)
            pids, scores = torch.tensor(pids)[scores_sorter.indices].tolist(), scores_sorter.values.tolist()

        return pids, scores


def batch(group, bsize, provide_offset=False):
    offset = 0
    while offset < len(group):
        L = group[offset: offset + bsize]
        yield ((offset, L) if provide_offset else L)
        offset += len(L)
    return


def load_queries(query_path):
    queries = {}
    for line in open(query_path):
        qid, text = line.split("\t")
        queries[qid] = text
    return queries


def retrieve(model_args: ModelArguments, data_args: DataArguments, eval_args: EvalArguments):
    config = AutoConfig.from_pretrained(model_args.backbone_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.backbone_name_or_path, config=config)
    model = AutoColBERTModel.from_pretrained(model_args.backbone_name_or_path, config=config)

    if model_args.adapter_name_or_path is not None:
        adapter_name = model.load_adapter(model_args.adapter_name_or_path)
        model.set_active_adapters(adapter_name)
        if model_args.merge_lora:
            model.merge_adapter(adapter_name)
        else:
            print("If your REM has LoRA modules, you can pass --merge_lora argument to merge LoRA weights and speed up inference.")
    
    model.to(eval_args.device)
    model.eval()
    inference = ModelInference(model, tokenizer=tokenizer, max_query_len=data_args.max_query_len, max_doc_len=None)
    ranker = Ranker(
        index_path = data_args.index_path, 
        faiss_index_path = data_args.faiss_index_path, 
        nprobe = eval_args.nprobe, 
        part_range = eval_args.part_range, 
        dim = model_args.output_dim, 
        inference = inference, 
        device = eval_args.device,
        faiss_depth=eval_args.faiss_depth
    )

    origin_docids = ujson.load(open(data_args.docids_path))
    milliseconds = 0

    with open(eval_args.output_ranking_path, 'w') as rlogger:
        queries = load_queries(data_args.query_path)
        qids_in_order = list(queries.keys())

        for qoffset, qbatch in batch(qids_in_order, 100, provide_offset=True):
            qbatch_text = [queries[qid] for qid in qbatch]

            rankings = []

            for query_idx, q in enumerate(qbatch_text):
                torch.cuda.synchronize(eval_args.device)
                s = time.time()

                Q = ranker.encode([q])
                pids, scores = ranker.rank(Q)

                torch.cuda.synchronize()
                milliseconds += (time.time() - s) * 1000.0

                if len(pids):
                    print(qoffset+query_idx, q, len(scores), len(pids), scores[0], pids[0],
                          milliseconds / (qoffset+query_idx+1), 'ms')

                rankings.append(zip(pids, scores))

            for query_idx, (qid, ranking) in enumerate(zip(qbatch, rankings)):
                query_idx = qoffset + query_idx

                if query_idx % 100 == 0:
                    print_message(f"#> Logging query #{query_idx} (qid {qid}) now...")

                ranking = [(score, pid, None) for pid, score in itertools.islice(ranking, eval_args.depth)]

                for rank, (score, pid, passage) in enumerate(ranking):
                    rlogger.write(f"{qid}\tQ0\t{origin_docids[pid]}\t{rank+1}\t{score}\tSystem\n")

    print("#> Done.")


def main():
    random.seed(12345)
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

    retrieve(model_args, data_args, eval_args)
    
    if data_args.qrel_path is not None:
        metric_scores = pytrec_evaluate(data_args.qrel_path, eval_args.output_ranking_path)
        for k in metric_scores.keys():
            if k != "perquery":
                print(metric_scores[k])
        ujson.dump(metric_scores, open(eval_args.output_metric_path, 'w'), indent=1)

if __name__ == "__main__":
    main()