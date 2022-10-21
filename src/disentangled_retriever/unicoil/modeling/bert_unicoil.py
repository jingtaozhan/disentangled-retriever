import torch
from transformers import BertAdapterModel


class BertUnicoil(BertAdapterModel):
    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, return_dict=False):
        outputs = super().forward(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids, 
            position_ids = position_ids,
            return_dict = True,
        )
        tok_weights = torch.relu(outputs.logits) * attention_mask.unsqueeze(-1).float()
        vocab_emb = torch.zeros(
            *input_ids.size(), self.config.vocab_size, 
            dtype=tok_weights.dtype, device=tok_weights.device
        )
        vocab_emb = torch.scatter(vocab_emb, dim=-1, index=input_ids.unsqueeze(-1), src=tok_weights)        
        vocab_emb = torch.max(vocab_emb, dim=1).values
        vocab_emb[:, [0, 101, 102, 103]] *= 0 # mask tokens: pad, unk, cls, sep
        if return_dict:
            outputs.logits = vocab_emb
            return outputs
        else:
            return vocab_emb
    
    def add_pooling_layer(self, head_name):
        self.add_tagging_head(head_name, num_labels=1, layers=1)


def test_output():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertUnicoil.from_pretrained("bert-base-uncased")
    model.add_pooling_layer("uni_pooling")
    print(model)

    queries  = ["When will the COVID-19 pandemic end?", "What are the impacts of COVID-19 pandemic to society?"]
    passages = ["It will end soon.", "It makes us care for each other."]
    query_embeds = model(**tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512))
    passage_embeds = model(**tokenizer(passages, return_tensors="pt", padding=True, truncation=True, max_length=512))

    print(query_embeds.shape)
    print(passage_embeds.shape)
    print(query_embeds @ passage_embeds.T)


def test_entire_save():
    model_init = BertUnicoil.from_pretrained("bert-base-uncased")
    model_init.add_pooling_layer("uni_pooling")
    print(model_init)
    model_init.save_pretrained("./data/temp")

    model_new = BertUnicoil.from_pretrained("./data/temp")
    print(model_new)


def test_disentangle_save():
    model_init = BertUnicoil.from_pretrained("bert-base-uncased")
    from transformers.adapters import ParallelConfig
    model_init.add_adapter("msmarco", config=ParallelConfig(reduction_factor=4))
    model_init.add_pooling_layer("msmarco")
    model_init.train_adapter("msmarco")
    model_init.save_all_adapters("./data/temp")
    model_init.save_all_heads("./data/temp")

    model_new = BertUnicoil.from_pretrained("bert-base-uncased")
    model_new.load_adapter("./data/temp/msmarco")
    print(model_new)


if __name__ == "__main__":
    test_disentangle_save()
