# Generic Relevance Estimation

## Unsupervisedly train DAM 

```bash
output_dir="./data/dense-mlm/chinese-dureader/train_rem/dam"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.adapt.run_adapt_with_mlm \
    --corpus_path ./data/datasets/dureader/corpus.tsv \
    --output_dir $output_dir \
    --model_name_or_path bert-base-chinese \
    --logging_first_step \
    --logging_steps 50 \
    --max_seq_length 256 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 1000 \
    --fp16 \
    --learning_rate 5e-5 \
    --max_steps 6000 \
    --dataloader_drop_last \
    --overwrite_output_dir \
    --dataloader_num_workers 16 \
    --weight_decay 0.01 \
    --lr_scheduler_type "constant_with_warmup" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --optim adamw_torch 
``` 

The trained model is actually our uploaded `jingtao/DAM-bert_base-mlm-dureader'. In the following, we will use this uploaded version to initialize the Transformer backbone. 

## Contrastively train REM 

```bash

output_dir="./data/dense-mlm/chinese-dureader/train_rem/rem-with-hf-dam/contrast"

python -m torch.distributed.launch --nproc_per_node 4 \ 
    -m disentangled_retriever.dense.finetune.run_contrast \
    --lora_rank 192 --parallel_reduction_factor 4 --new_adapter_name dureader \
    --pooling average \
    --similarity_metric cos \
    --qrel_path ./data/datasets/dureader/qrels.train \
    --query_path ./data/datasets/dureader/query.train \
    --corpus_path ./data/datasets/dureader/corpus.tsv \
    --output_dir $output_dir \
    --model_name_or_path jingtao/DAM-bert_base-mlm-dureader \
    --logging_steps 10 \
    --max_query_len 24 \
    --max_doc_len 384 \
    --per_device_train_batch_size 32 \
    --inv_temperature 20 \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --neg_per_query 0 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --dataloader_drop_last \
    --overwrite_output_dir \
    --dataloader_num_workers 0 \
    --weight_decay 0 \
    --lr_scheduler_type "constant" \
    --save_strategy "epoch" \
    --optim adamw_torch

```

Now evaluate the out-of-domain performance with CPR-Ecommerce.
```bash
data_dir="./data/datasets/cpr-ecom"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-dureader-cpr_ecom"
adapter_name_or_path="./data/dense-mlm/chinese-dureader/train_rem/rem-with-hf-dam/contrast/dureader"
output_dir="./data/dense-mlm/chinese-dureader/train_rem/rem-with-hf-dam/contrast/evaluate/cpr-ecom"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.evaluate.run_eval \
    --backbone_name_or_path $backbone_name_or_path \
    --adapter_name_or_path $adapter_name_or_path \
    --corpus_path $data_dir/corpus.tsv \
    --query_path $data_dir/query.dev \
    --qrel_path $data_dir/qrels.dev \
    --output_dir $output_dir \
    --out_corpus_dir $output_dir/corpus \
    --out_query_dir $output_dir/dev \
    --per_device_eval_batch_size 48 \
    --overwrite_output_dir \
    --dataloader_num_workers 8 \
    --pooling average \
    --similarity_metric cos \
    --topk 1000 \
    --merge_lora
```

The results are
```python
{'NDCG@1': 0.199, 'NDCG@3': 0.26921, 'NDCG@5': 0.28901, 'NDCG@10': 0.31223, 'NDCG@100': 0.36534}
{'MAP@1': 0.199, 'MAP@3': 0.252, 'MAP@5': 0.263, 'MAP@10': 0.27254, 'MAP@100': 0.28207}
{'Recall@10': 0.439, 'Recall@50': 0.618, 'Recall@100': 0.703, 'Recall@200': 0.769, 'Recall@500': 0.832, 'Recall@1000': 0.879}
{'P@1': 0.199, 'P@3': 0.10633, 'P@5': 0.0734, 'P@10': 0.0439, 'P@100': 0.00703}
{'MRR@10': 0.27254, 'MRR@100': 0.28207}
```
The results slightly differ from the reported results in our paper, largely due to different environments.
