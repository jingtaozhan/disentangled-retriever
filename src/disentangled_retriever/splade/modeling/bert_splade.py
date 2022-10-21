import torch
from transformers import BertForMaskedLM


class BertSplade(BertForMaskedLM):
    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, return_dict=False):
        outputs = super().forward(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids, 
            position_ids = position_ids,
            return_dict = True,
        )
        vocab_emb, _ = torch.max(torch.log(1 + torch.relu(outputs.logits)) * attention_mask.unsqueeze(-1), dim=1)
        if return_dict:
            outputs.logits = vocab_emb
            return outputs
        else:
            return vocab_emb


def test_output():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertSplade.from_pretrained("bert-base-uncased")

    queries  = ["When will the COVID-19 pandemic end?", "What are the impacts of COVID-19 pandemic to society?"]
    passages = ["It will end soon.", "It makes us care for each other."]
    query_embeds = model(**tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512))
    passage_embeds = model(**tokenizer(passages, return_tensors="pt", padding=True, truncation=True, max_length=512))

    print(query_embeds.shape)
    print(passage_embeds.shape)
    print(query_embeds @ passage_embeds.T)


def test_disentangle_save():
    import os
    from transformers import AutoTokenizer
    model_init = BertSplade.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    from transformers.adapters import ParallelConfig
    print(model_init.bert.embeddings.word_embeddings._parameters['weight'])
    # model_init.add_embeddings("msmarco", tokenizer)
    # model_init.set_output_embeddings(model_init.get_input_embeddings())
    # model_init.add_embeddings("msmarco", tokenizer, reference_embedding="default", reference_tokenizer=tokenizer)
    model_init.add_adapter("msmarco", config=ParallelConfig(reduction_factor=4))
    model_init.train_adapter("msmarco", train_embeddings=True)
    for param in model_init.get_input_embeddings().parameters():
        param.requires_grad = True
    model_init.save_all_adapters("./data/temp")
    # model_init.save_embeddings(os.path.join("./data/temp", model_init.active_embeddings), model_init.active_embeddings)

    model_new = BertSplade.from_pretrained("jingtao/DAM-bert_base-mlm-msmarco")
    # model_new.load_embeddings("./data/temp/msmarco", "new")
    print(model_new.bert.embeddings.word_embeddings._parameters['weight'])
    model_new.load_adapter("./data/temp/msmarco")
    
    # print(model_init.cls.predictions.transform.dense._parameters['weight'])
    # print(model_new.cls.predictions.transform.dense._parameters['weight'])

    print(model_init.bert.embeddings.word_embeddings._parameters['weight'])
    print(model_init.cls.predictions.decoder._parameters['weight'])
    print(model_new.bert.embeddings.word_embeddings._parameters['weight'])
    print(model_new.cls.predictions.decoder._parameters['weight'])


def test_disentangle_train():
    from transformers import AutoTokenizer
    model = BertSplade.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    from transformers.adapters import ParallelConfig
    model_param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.add_adapter("msmarco", config=ParallelConfig(reduction_factor=4))

    model.train_adapter("msmarco", train_embeddings=True)
    for param in model.get_input_embeddings().parameters():
        param.requires_grad = True
    print(f"Parameters with gradient: {sorted([(n, p.shape) for n, p in model.named_parameters() if p.requires_grad])}")
    adapter_param_cnt = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    print(f"adapter_param_cnt:{adapter_param_cnt}, model_param_cnt:{model_param_cnt}, ratio:{adapter_param_cnt/model_param_cnt:.4f}")


if __name__ == "__main__":
    test_disentangle_save()
