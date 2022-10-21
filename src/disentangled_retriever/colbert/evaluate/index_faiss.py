import os
import sys
import math
import time
import ujson
import queue
import faiss
import torch
import datetime
import itertools
import threading
import numpy as np
import argparse


def load_doclens(directory, flatten=True):
    parts, _, _ = get_parts(directory)

    doclens_filenames = [os.path.join(directory, 'doclens.{}.json'.format(filename)) for filename in parts]
    all_doclens = [ujson.load(open(filename)) for filename in doclens_filenames]

    if flatten:
        all_doclens = [x for sub_doclens in all_doclens for x in sub_doclens]

    return all_doclens


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
        Example: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
        Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """

    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def print_message(*s, condition=True):
    s = ' '.join([str(x) for x in s])
    msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

    if condition:
        print(msg, flush=True)

    return msg


class FaissIndexGPU():
    def __init__(self):
        self.ngpu = faiss.get_num_gpus()

        if self.ngpu == 0:
            return

        self.tempmem = 1 << 33
        self.max_add_per_gpu = 1 << 25
        self.max_add = self.max_add_per_gpu * self.ngpu
        self.add_batch_size = 65536

        self.gpu_resources = self._prepare_gpu_resources()

    def _prepare_gpu_resources(self):
        print_message(f"Preparing resources for {self.ngpu} GPUs.")

        gpu_resources = []

        for _ in range(self.ngpu):
            res = faiss.StandardGpuResources()
            if self.tempmem >= 0:
                res.setTempMemory(self.tempmem)
            gpu_resources.append(res)

        return gpu_resources

    def _make_vres_vdev(self):
        """
        return vectors of device ids and resources useful for gpu_multiple
        """

        assert self.ngpu > 0

        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()

        for i in range(self.ngpu):
            vdev.push_back(i)
            vres.push_back(self.gpu_resources[i])

        return vres, vdev

    def training_initialize(self, index, quantizer):
        """
        The index and quantizer should be owned by caller.
        """
        assert self.ngpu > 0
        s = time.time()
        self.index_ivf = faiss.extract_index_ivf(index)
        self.clustering_index = faiss.index_cpu_to_all_gpus(quantizer)
        self.index_ivf.clustering_index = self.clustering_index
        print(time.time() - s)

    def training_finalize(self):
        assert self.ngpu > 0

        s = time.time()
        self.index_ivf.clustering_index = faiss.index_gpu_to_cpu(self.index_ivf.clustering_index)
        print(time.time() - s)

    def adding_initialize(self, index):
        """
        The index should be owned by caller.
        """

        assert self.ngpu > 0

        self.co = faiss.GpuMultipleClonerOptions()
        self.co.useFloat16 = True
        self.co.useFloat16CoarseQuantizer = False
        self.co.usePrecomputed = False
        self.co.indicesOptions = faiss.INDICES_CPU
        self.co.verbose = True
        self.co.reserveVecs = self.max_add
        self.co.shard = True
        assert self.co.shard_type in (0, 1, 2)

        self.vres, self.vdev = self._make_vres_vdev()
        self.gpu_index = faiss.index_cpu_to_gpu_multiple(self.vres, self.vdev, index, self.co)

    def add(self, index, data, offset):
        assert self.ngpu > 0

        t0 = time.time()
        nb = data.shape[0]

        for i0 in range(0, nb, self.add_batch_size):
            i1 = min(i0 + self.add_batch_size, nb)
            xs = data[i0:i1]

            self.gpu_index.add_with_ids(xs, np.arange(offset+i0, offset+i1))

            if self.max_add > 0 and self.gpu_index.ntotal > self.max_add:
                self._flush_to_cpu(index, nb, offset)

            print('\r%d/%d (%.3f s)  ' % (i0, nb, time.time() - t0), end=' ')
            sys.stdout.flush()

        if self.gpu_index.ntotal > 0:
            self._flush_to_cpu(index, nb, offset)

        assert index.ntotal == offset+nb, (index.ntotal, offset+nb, offset, nb)
        print(f"add(.) time: %.3f s \t\t--\t\t index.ntotal = {index.ntotal}" % (time.time() - t0))

    def _flush_to_cpu(self, index, nb, offset):
        print("Flush indexes to CPU")

        for i in range(self.ngpu):
            index_src_gpu = faiss.downcast_index(self.gpu_index if self.ngpu == 1 else self.gpu_index.at(i))
            index_src = faiss.index_gpu_to_cpu(index_src_gpu)

            index_src.copy_subset_to(index, 0, offset, offset+nb)
            index_src_gpu.reset()
            index_src_gpu.reserveMemory(self.max_add)

        if self.ngpu > 1:
            try:
                self.gpu_index.sync_with_shard_indexes()
            except:
                self.gpu_index.syncWithSubIndexes()


class FaissIndex():
    def __init__(self, dim, partitions):
        self.dim = dim
        self.partitions = partitions

        self.gpu = FaissIndexGPU()
        self.quantizer, self.index = self._create_index()
        self.offset = 0

    def _create_index(self):
        quantizer = faiss.IndexFlatL2(self.dim)  # faiss.IndexHNSWFlat(dim, 32)
        index = faiss.IndexIVFPQ(quantizer, self.dim, self.partitions, 16, 8)

        return quantizer, index

    def train(self, train_data):
        print_message(f"#> Training now (using {self.gpu.ngpu} GPUs)...")

        if self.gpu.ngpu > 0:
            self.gpu.training_initialize(self.index, self.quantizer)

        s = time.time()
        self.index.train(train_data)
        print(time.time() - s)

        if self.gpu.ngpu > 0:
            self.gpu.training_finalize()

    def add(self, data):
        print_message(f"Add data with shape {data.shape} (offset = {self.offset})..")

        if self.gpu.ngpu > 0 and self.offset == 0:
            self.gpu.adding_initialize(self.index)

        if self.gpu.ngpu > 0:
            self.gpu.add(self.index, data, self.offset)
        else:
            self.index.add(data)

        self.offset += data.shape[0]

    def save(self, output_path):
        print_message(f"Writing index to {output_path} ...")

        self.index.nprobe = 10  # just a default
        faiss.write_index(self.index, output_path)



def get_parts(directory):
    extension = '.pt'

    parts = sorted([int(filename[: -1 * len(extension)]) for filename in os.listdir(directory)
                    if filename.endswith(extension)])

    assert list(range(len(parts))) == parts, parts

    # Integer-sortedness matters.
    parts_paths = [os.path.join(directory, '{}{}'.format(filename, extension)) for filename in parts]
    samples_paths = [os.path.join(directory, '{}.sample'.format(filename)) for filename in parts]

    return parts, parts_paths, samples_paths


def load_index_part(filename, verbose=True):
    part = torch.load(filename)

    if type(part) == list:  # for backward compatibility
        part = torch.cat(part)

    return part


# def get_faiss_index_name(args, offset=None, endpos=None):
#     partitions_info = '' if args.partitions is None else f'.{args.partitions}'
#     range_info = '' if offset is None else f'.{offset}-{endpos}'

#     return f'ivfpq{partitions_info}{range_info}.faiss'


def load_sample(samples_paths, sample_fraction=None):
    sample = []

    for filename in samples_paths:
        print_message(f"#> Loading {filename} ...")
        part = load_index_part(filename)
        if sample_fraction:
            part = part[torch.randint(0, high=part.size(0), size=(int(part.size(0) * sample_fraction),))]
        sample.append(part)

    sample = torch.cat(sample).float().numpy()

    print("#> Sample has shape", sample.shape)

    return sample


def prepare_faiss_index(slice_samples_paths, partitions, sample_fraction=None):
    training_sample = load_sample(slice_samples_paths, sample_fraction=sample_fraction)

    dim = training_sample.shape[-1]
    index = FaissIndex(dim, partitions)

    print_message("#> Training with the vectors...")

    index.train(training_sample)

    print_message("Done training!\n")

    return index


SPAN = 3


def index_faiss(args):
    print_message("#> Starting..")

    parts, parts_paths, samples_paths = get_parts(args.index_path)

    if args.sample is not None:
        assert args.sample, args.sample
        print_message(f"#> Training with {round(args.sample * 100.0, 1)}% of *all* embeddings (provided --sample).")
        samples_paths = parts_paths

    num_parts_per_slice = math.ceil(len(parts) / args.slices)

    for slice_idx, part_offset in enumerate(range(0, len(parts), num_parts_per_slice)):
        part_endpos = min(part_offset + num_parts_per_slice, len(parts))

        slice_parts_paths = parts_paths[part_offset:part_endpos]
        slice_samples_paths = samples_paths[part_offset:part_endpos]

        # if args.slices == 1:
        #     faiss_index_name = get_faiss_index_name(args)
        # else:
        #     faiss_index_name = get_faiss_index_name(args, offset=part_offset, endpos=part_endpos)

        # output_path = os.path.join(args.index_path, faiss_index_name)
        output_path = os.path.join(args.index_path, "ivfpq.faiss")
        print_message(f"#> Processing slice #{slice_idx+1} of {args.slices} (range {part_offset}..{part_endpos}).")
        print_message(f"#> Will write to {output_path}.")

        assert not os.path.exists(output_path), output_path

        index = prepare_faiss_index(slice_samples_paths, args.partitions, args.sample)

        loaded_parts = queue.Queue(maxsize=1)

        def _loader_thread(thread_parts_paths):
            for filenames in grouper(thread_parts_paths, SPAN, fillvalue=None):
                sub_collection = [load_index_part(filename) for filename in filenames if filename is not None]
                sub_collection = torch.cat(sub_collection)
                sub_collection = sub_collection.float().numpy()
                loaded_parts.put(sub_collection)

        thread = threading.Thread(target=_loader_thread, args=(slice_parts_paths,))
        thread.start()

        print_message("#> Indexing the vectors...")

        for filenames in grouper(slice_parts_paths, SPAN, fillvalue=None):
            print_message("#> Loading", filenames, "(from queue)...")
            sub_collection = loaded_parts.get()

            print_message("#> Processing a sub_collection with shape", sub_collection.shape)
            index.add(sub_collection)

        print_message("Done indexing!")

        index.save(output_path)

        print_message(f"\n\nDone! All complete (for slice #{slice_idx+1} of {args.slices})!")

        thread.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--partitions", type=int, default=None)
    parser.add_argument('--sample', dest='sample', default=None, type=float)
    parser.add_argument('--slices', dest='slices', default=1, type=int)
    args = parser.parse_args()

    assert args.slices >= 1
    assert args.sample is None or (0.0 < args.sample < 1.0), args.sample

    num_embeddings = sum(load_doclens(args.index_path))
    print("#> num_embeddings =", num_embeddings)

    if args.partitions is None:
        args.partitions = 1 << math.ceil(math.log2(8 * math.sqrt(num_embeddings)))
        print('\n\n')
        print("You did not specify --partitions!")
        print("Default computation chooses", args.partitions,
                    "partitions (for {} embeddings)".format(num_embeddings))
        print('\n\n')

    index_faiss(args)


if __name__ == "__main__":
    main()