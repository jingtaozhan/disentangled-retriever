from transformers import DistilBertAdapterModel

from .utils import extract_text_embed


class DistilBertDense(DistilBertAdapterModel):
    def forward(self, input_ids, attention_mask, return_dict=False):
        pooling = getattr(self.config, "pooling")
        outputs = self.distilbert(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            return_dict = True,
        )
        # Most models in the literature use IP and cls-pooling
        similarity_metric = getattr(self.config, "similarity_metric")
        text_embeds = extract_text_embed(
            last_hidden_state = outputs.last_hidden_state, 
            attention_mask = attention_mask,
            similarity_metric = similarity_metric, 
            pooling = pooling,
        )
        if return_dict:
            outputs.embedding = text_embeds
            return outputs
        else:
            return text_embeds