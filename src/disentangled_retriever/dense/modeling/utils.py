import torch
from torch import Tensor
import torch.nn.functional as F


SIMILARITY_METRIC_IP = "ip"
SIMILARITY_METRIC_COS = "cos"
SIMILARITY_METRICS = [SIMILARITY_METRIC_IP, SIMILARITY_METRIC_COS]

POOLING_AVERAGE = "average"
POOLING_CLS = "cls"
POOLING_METHODS = [POOLING_AVERAGE, POOLING_CLS]


def extract_text_embed(
        last_hidden_state: Tensor, 
        attention_mask: Tensor, 
        similarity_metric: str, 
        pooling: str
    ):
    if pooling == POOLING_CLS:
        text_embeds = last_hidden_state[:, 0]
    elif pooling == POOLING_AVERAGE:
        masked_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.)
        text_embeds = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    else:
        raise NotImplementedError()
    if similarity_metric == SIMILARITY_METRIC_IP:
        pass
    elif similarity_metric == SIMILARITY_METRIC_COS:
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    else:
        raise NotImplementedError()
    return text_embeds