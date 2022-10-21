import torch
from transformers import BertAdapterModel


class ColBERT(BertAdapterModel):
    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, return_dict=False):
        outputs = super().forward(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids, 
            position_ids = position_ids,
            return_dict = True,
        )
        outputs.logits = torch.nn.functional.normalize(outputs.logits, p=2, dim=2)
        outputs.logits *= attention_mask[:, :, None].float()
        # TODO The original implememntation of ColBERT also masks punctuations for docs. Necessary?
        if return_dict:
            return outputs
        else:
            return outputs.logits
    
    def add_pooling_layer(self, head_name, out_dim):
        self.add_tagging_head(head_name, num_labels=out_dim, layers=1)

    @staticmethod
    def compute_similarity(q_reps, p_reps):
        token_scores = torch.einsum('qin,pjn->qipj', q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = scores.sum(1)
        return scores


def test_output():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = ColBERT.from_pretrained("bert-base-uncased")
    model.add_pooling_layer("col_pooling", 32)
    print(model)

    queries  = ["When will the COVID-19 pandemic end?", "What are the impacts of COVID-19 pandemic to society?"]
    passages = ["It will end soon.", "It makes us care for each other."]
    query_embeds = model(**tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512))
    passage_embeds = model(**tokenizer(passages, return_tensors="pt", padding=True, truncation=True, max_length=512))

    print(query_embeds.shape)
    print(passage_embeds.shape)
    print(model.compute_similarity(query_embeds, passage_embeds))


def test_entire_save():
    model_init = ColBERT.from_pretrained("bert-base-uncased")
    model_init.add_pooling_layer("col_pooling", 32)
    print(model_init)
    model_init.save_pretrained("./data/temp")

    model_new = ColBERT.from_pretrained("./data/temp")
    print(model_new)


def test_disentangle_save():
    model_init = ColBERT.from_pretrained("bert-base-uncased")
    model_init.add_pooling_layer("msmarco", 32)
    from transformers.adapters import ParallelConfig
    model_init.add_adapter("msmarco", config=ParallelConfig(reduction_factor=4))
    model_init.train_adapter("msmarco")
    model_init.save_all_adapters("./data/temp")
    model_init.save_all_heads("./data/temp")

    model_new = ColBERT.from_pretrained("bert-base-uncased")
    model_new.load_adapter("./data/temp/msmarco")
    print(model_new)


if __name__ == "__main__":
    test_disentangle_save()
