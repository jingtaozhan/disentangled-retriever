import torch
from tqdm import tqdm


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))

    return batches


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def tensorize_queries(tokenizer, batch_text, max_query_len, bsize=None):
    assert type(batch_text) in [list, tuple], (type(batch_text))

    obj = tokenizer(
        batch_text,
        padding=True,
        return_tensors='pt',
        add_special_tokens=True,
        return_attention_mask=True,
        truncation=True,
        max_length=max_query_len
    )

    ids, mask = obj['input_ids'], obj['attention_mask']

    if bsize:
        batches = _split_into_batches(ids, mask, bsize)
        return batches

    return ids, mask


def tensorize_docs(tokenizer, batch_text, max_doc_len, bsize=None):
    assert type(batch_text) in [list, tuple], (type(batch_text))

    obj = tokenizer(
        batch_text,
        padding=True,
        return_tensors='pt',
        add_special_tokens=True,
        return_attention_mask=True,
        truncation=True,
        max_length=max_doc_len
    )

    ids, mask = obj['input_ids'], obj['attention_mask']

    if bsize:
        ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
        batches = _split_into_batches(ids, mask, bsize)
        return batches, reverse_indices

    return ids, mask


class ModelInference():
    def __init__(self, colbert, tokenizer, max_query_len, max_doc_len):
        self.colbert = colbert
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

    def query(self, input_ids, attention_mask, to_cpu=False):
        input_ids, attention_mask = input_ids.to(self.colbert.device), attention_mask.to(self.colbert.device)
        with torch.no_grad():
            Q = self.colbert(input_ids=input_ids, attention_mask=attention_mask)
            return Q.cpu() if to_cpu else Q

    def doc(self, input_ids, attention_mask, keep_dims, to_cpu=False):
        input_ids, attention_mask = input_ids.to(self.colbert.device), attention_mask.to(self.colbert.device)
        with torch.no_grad():
            D = self.colbert(input_ids=input_ids, attention_mask=attention_mask)

            if not keep_dims:
                D, mask = D.cpu().to(dtype=torch.float16), attention_mask.cpu().bool().squeeze(-1)
                D = [d[mask[idx]] for idx, d in enumerate(D)]

            return D.cpu() if to_cpu else D

    def queryFromText(self, queries, bsize=None, to_cpu=False):
        if bsize:
            batches = tensorize_queries(self.tokenizer, queries, self.max_query_len, bsize=bsize)
            batches = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in batches]
            return torch.cat(batches)

        input_ids, attention_mask = tensorize_queries(self.tokenizer, queries, self.max_query_len)
        return self.query(input_ids, attention_mask)

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False):
        if bsize:
            batches, reverse_indices = tensorize_docs(self.tokenizer, docs, self.max_query_len, bsize=bsize)

            batches = [self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)
                       for input_ids, attention_mask in batches]

            if keep_dims:
                D = _stack_3D_tensors(batches)
                return D[reverse_indices]

            D = [d for batch in batches for d in batch]
            return [D[idx] for idx in reverse_indices.tolist()]

        input_ids, attention_mask = tensorize_docs(self.tokenizer, docs, self.max_query_len)
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims)


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos

    return output
