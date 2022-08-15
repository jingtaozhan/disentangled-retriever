# Generic Relevance Estimation

## Unsupervisedly train DAM 

```bash
output_dir="./data/dense-mlm/english-msmarco/train_rem/dam"

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

output_dir="./data/dense-mlm/english-msmarco/train_rem/rem-with-hf-dam/contrast"

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
adapter_name_or_path="./data/dense-mlm/english-msmarco/train_rem/rem-with-hf-dam/contrast/msmarco"
output_dir="./data/dense-mlm/english-msmarco/train_rem/rem-with-hf-dam/contrast/evaluate/lotte/technology/test"

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
```

## Distilled REM


```bash
output_dir="./data/dense-mlm/english-msmarco/train_rem/rem-with-hf-dam/distil"

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
adapter_name_or_path="./data/dense-mlm/english-msmarco/train_rem/rem-with-hf-dam/distil/msmarco"
output_dir="./data/dense-mlm/english-msmarco/train_rem/rem-with-hf-dam/distil/evaluate/lotte/technology/test"

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
```