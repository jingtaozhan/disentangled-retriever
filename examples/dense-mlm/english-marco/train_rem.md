# Generic Relevance Estimation

## Unsupervisedly train DAM 

```bash
output_dir="./data/dense-mlm/english-marco/train_rem/dam"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.adapt.run_adapt_with_mlm \
    --corpus_path ./data/datasets/msmarco-passage/corpus.tsv \
    --output_dir $output_dir \
    --model_name_or_path bert-base-uncased \
    --logging_first_step \
    --logging_steps 50 \
    --max_seq_length 100 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 1000 \
    --fp16 \
    --learning_rate 5e-5 \
    --max_steps 100000 \
    --dataloader_drop_last \
    --overwrite_output_dir \
    --dataloader_num_workers 16 \
    --weight_decay 0.01 \
    --lr_scheduler_type "constant_with_warmup" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --optim adamw_torch 
``` 

The trained model is actually our uploaded `jingtao/DAM-bert_base-mlm-msmarco'. In the following, we will use this uploaded version to initialize the Transformer backbone. 

## Contrastively train REM 

```bash

output_dir="./data/dense-mlm/english-marco/train_rem/rem-with-hf-dam/contrast"

python -m torch.distributed.launch --nproc_per_node 4 \ 
    -m disentangled_retriever.dense.finetune.run_contrast \
    --lora_rank 192 --parallel_reduction_factor 4 --new_adapter_name msmarco \
    --pooling average \
    --similarity_metric ip \
    --qrel_path ./data/datasets/msmarco-passage/qrels.train \
    --query_path ./data/datasets/msmarco-passage/query.train \
    --corpus_path ./data/datasets/msmarco-passage/corpus.tsv \
    --negative ./data/datasets/msmarco-passage/msmarco-hard-negatives.tsv \
    --output_dir $output_dir \
    --model_name_or_path jingtao/DAM-bert_base-mlm-msmarco \
    --logging_steps 100 \
    --max_query_len 24 \
    --max_doc_len 128 \
    --per_device_train_batch_size 32 \
    --inv_temperature 1 \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --neg_per_query 3 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --dataloader_drop_last \
    --overwrite_output_dir \
    --dataloader_num_workers 0 \
    --weight_decay 0 \
    --lr_scheduler_type "constant" \
    --save_strategy "epoch" \
    --optim adamw_torch

```

Now evaluate the out-of-domain performance with Lotte-Technology.
```bash
data_dir="./data/datasets/lotte/technology/test"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-msmarco-lotte_tech_test"
adapter_name_or_path="./data/dense-mlm/english-marco/train_rem/rem-with-hf-dam/contrast/msmarco"
output_dir="./data/dense-mlm/english-marco/train_rem/rem-with-hf-dam/contrast/evaluate/lotte/technology/test"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.evaluate.run_eval \
    --backbone_name_or_path $backbone_name_or_path \
    --adapter_name_or_path $adapter_name_or_path \
    --corpus_path $data_dir/corpus.tsv \
    --query_path $data_dir/query.forum \
    --qrel_path $data_dir/qrels.forum \
    --output_dir $output_dir \
    --out_corpus_dir $output_dir/corpus \
    --out_query_dir $output_dir/forum \
    --per_device_eval_batch_size 48 \
    --overwrite_output_dir \
    --dataloader_num_workers 8 \
    --pooling average \
    --similarity_metric ip \
    --topk 1000 \
    --merge_lora
```

The results are
```python
{'NDCG@1': 0.24451, 'NDCG@3': 0.21149, 'NDCG@5': 0.19755, 'NDCG@10': 0.19727, 'NDCG@100': 0.27316}
{'MAP@1': 0.04264, 'MAP@3': 0.0783, 'MAP@5': 0.09308, 'MAP@10': 0.10892, 'MAP@100': 0.13237}
{'Recall@10': 0.19013, 'Recall@50': 0.34759, 'Recall@100': 0.43041, 'Recall@200': 0.51139, 'Recall@500': 0.62014, 'Recall@1000': 0.69698}
{'P@1': 0.24451, 'P@3': 0.19411, 'P@5': 0.16088, 'P@10': 0.11756, 'P@100': 0.02876}
{'MRR@10': 0.34878, 'MRR@100': 0.35952}
```
Performance scores on Lotte-Writing are
```python
{'NDCG@1': 0.5075, 'NDCG@3': 0.44851, 'NDCG@5': 0.43455, 'NDCG@10': 0.44481, 'NDCG@100': 0.51248}
{'MAP@1': 0.122, 'MAP@3': 0.23524, 'MAP@5': 0.28597, 'MAP@10': 0.32969, 'MAP@100': 0.36022}
{'Recall@10': 0.45208, 'Recall@50': 0.60957, 'Recall@100': 0.66042, 'Recall@200': 0.70782, 'Recall@500': 0.76716, 'Recall@1000': 0.80584}
{'P@1': 0.5075, 'P@3': 0.402, 'P@5': 0.3312, 'P@10': 0.2253, 'P@100': 0.03567}
{'MRR@10': 0.61241, 'MRR@100': 0.61769}
```
The results slightly differ from the reported results in our paper, largely due to different environments.


## Distilled REM


```bash
output_dir="./data/dense-mlm/english-marco/train_rem/rem-with-hf-dam/distil"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.finetune.run_distill \
    --lora_rank 192 --parallel_reduction_factor 4 --new_adapter_name msmarco \
    --pooling average \
    --similarity_metric ip \
    --qrel_path ./data/datasets/msmarco-passage/qrels.train \
    --query_path ./data/datasets/msmarco-passage/query.train \
    --corpus_path ./data/datasets/msmarco-passage/corpus.tsv \
    --ce_scores_file ./data/datasets/msmarco-passage/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz \
    --output_dir $output_dir \
    --model_name_or_path jingtao/DAM-bert_base-mlm-msmarco \
    --logging_steps 100 \
    --max_query_len 24 \
    --max_doc_len 128 \
    --per_device_train_batch_size 32 \
    --inv_temperature 1 \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --neg_per_query 3 \
    --learning_rate 2e-5 \
    --num_train_epochs 20 \
    --dataloader_drop_last \
    --overwrite_output_dir \
    --dataloader_num_workers 0 \
    --weight_decay 0 \
    --lr_scheduler_type "constant" \
    --save_strategy "epoch" \
    --optim adamw_torch
```


Now evaluate the out-of-domain performance with Lotte-Technology.
```bash
data_dir="./data/datasets/lotte/technology/test"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-msmarco-lotte_tech_test"
adapter_name_or_path="./data/dense-mlm/english-marco/train_rem/rem-with-hf-dam/distil/msmarco"
output_dir="./data/dense-mlm/english-marco/train_rem/rem-with-hf-dam/distil/evaluate/lotte/technology/test"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.evaluate.run_eval \
    --backbone_name_or_path $backbone_name_or_path \
    --adapter_name_or_path $adapter_name_or_path \
    --corpus_path $data_dir/corpus.tsv \
    --query_path $data_dir/query.forum \
    --qrel_path $data_dir/qrels.forum \
    --output_dir $output_dir \
    --out_corpus_dir $output_dir/corpus \
    --out_query_dir $output_dir/forum \
    --per_device_eval_batch_size 48 \
    --overwrite_output_dir \
    --dataloader_num_workers 8 \
    --pooling average \
    --similarity_metric ip \
    --topk 1000 \
    --merge_lora
```

The results are
```python
{'NDCG@1': 0.25449, 'NDCG@3': 0.21583, 'NDCG@5': 0.2049, 'NDCG@10': 0.20679, 'NDCG@100': 0.28285}
{'MAP@1': 0.04597, 'MAP@3': 0.08256, 'MAP@5': 0.09892, 'MAP@10': 0.11601, 'MAP@100': 0.14065}
{'Recall@10': 0.2024, 'Recall@50': 0.36043, 'Recall@100': 0.43759, 'Recall@200': 0.51673, 'Recall@500': 0.62043, 'Recall@1000': 0.69662}
{'P@1': 0.25449, 'P@3': 0.19677, 'P@5': 0.16687, 'P@10': 0.12255, 'P@100': 0.02902}
{'MRR@10': 0.3593, 'MRR@100': 0.36983}
```
Performance scores on Lotte-Writing are
```python
{'NDCG@1': 0.5045, 'NDCG@3': 0.43784, 'NDCG@5': 0.42469, 'NDCG@10': 0.43442, 'NDCG@100': 0.50566}
{'MAP@1': 0.12039, 'MAP@3': 0.22904, 'MAP@5': 0.27773, 'MAP@10': 0.31979, 'MAP@100': 0.3518}
{'Recall@10': 0.44169, 'Recall@50': 0.60692, 'Recall@100': 0.65855, 'Recall@200': 0.69967, 'Recall@500': 0.75116, 'Recall@1000': 0.78384}
{'P@1': 0.5045, 'P@3': 0.39183, 'P@5': 0.3232, 'P@10': 0.2184, 'P@100': 0.03533}
{'MRR@10': 0.60539, 'MRR@100': 0.61146}
```
The results slightly differ from the reported results in our paper, largely due to different environments.
