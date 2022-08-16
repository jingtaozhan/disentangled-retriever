# Train Dense Retrieval on MS MARCO

## Contrastive Finetuning

Here is the training command. We find model performance convergences after five epochs, and here we set the maximum training epoch to $5$. The full batch size is $4 \times 32 = 128$. `neg_per_query' is set to $3$.  

```bash
output_dir="./data/dense-mlm/english-marco/baseline_train/contrast"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.finetune.run_contrast \
    --pooling average \
    --similarity_metric ip \
    --qrel_path ./data/datasets/msmarco-passage/qrels.train \
    --query_path ./data/datasets/msmarco-passage/query.train \
    --corpus_path ./data/datasets/msmarco-passage/corpus.tsv \
    --negative ./data/datasets/msmarco-passage/msmarco-hard-negatives.tsv \
    --output_dir $output_dir \
    --model_name_or_path bert-base-uncased \
    --logging_steps 100 \
    --max_query_len 24 \
    --max_doc_len 128 \
    --per_device_train_batch_size 32 \
    --inv_temperature 1 \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --neg_per_query 3 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --dataloader_drop_last \
    --overwrite_output_dir \
    --dataloader_num_workers 0 \
    --weight_decay 0 \
    --lr_scheduler_type "constant" \
    --save_strategy "epoch" \
    --optim adamw_torch
```


Here is an example code to evaluate the out-of-domain performance on Lotte-Tech.
```bash
data_dir="./data/datasets/lotte/technology/test"
backbone_name_or_path="./data/dense-mlm/english-marco/baseline_train/contrast"
output_dir="$backbone_name_or_path/evaluate/lotte/technology/test"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.evaluate.run_eval \
    --backbone_name_or_path $backbone_name_or_path \
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
    --topk 1000
```

Following the above commands, we train and evaluate a baseline model.
The evaluation results are
```python
{'NDCG@1': 0.17465, 'NDCG@3': 0.14778, 'NDCG@5': 0.13916, 'NDCG@10': 0.13832, 'NDCG@100': 0.19323}
{'MAP@1': 0.03004, 'MAP@3': 0.05305, 'MAP@5': 0.06324, 'MAP@10': 0.07282, 'MAP@100': 0.08777}
{'Recall@10': 0.13417, 'Recall@50': 0.25031, 'Recall@100': 0.30958, 'Recall@200': 0.37117, 'Recall@500': 0.45639, 'Recall@1000': 0.52626}
{'P@1': 0.17465, 'P@3': 0.13423, 'P@5': 0.11297, 'P@10': 0.08109, 'P@100': 0.02009}
{'MRR@10': 0.25906, 'MRR@100': 0.26972}
```
The results slightly differ from the reported results in our paper, largely due to different environments.


## Knowledge Distillation

Here is the training command. We find model performance convergences after twenty epochs, and here we set the maximum training epoch to $20$. The full batch size is $4 \times 32 = 128$. `neg_per_query' is set to $3$. Loss function is Marge-MSE.

```bash
output_dir="./data/dense-mlm/english-marco/baseline_train/distil"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.finetune.run_distill \
    --pooling average \
    --similarity_metric ip \
    --qrel_path ./data/datasets/msmarco-passage/qrels.train \
    --query_path ./data/datasets/msmarco-passage/query.train \
    --corpus_path ./data/datasets/msmarco-passage/corpus.tsv \
    --ce_scores_file ./data/datasets/msmarco-passage/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz \
    --output_dir $output_dir \
    --model_name_or_path bert-base-uncased \
    --logging_steps 100 \
    --max_query_len 24 \
    --max_doc_len 128 \
    --per_device_train_batch_size 32 \
    --inv_temperature 1 \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --neg_per_query 3 \
    --learning_rate 1e-5 \
    --num_train_epochs 20 \
    --dataloader_drop_last \
    --overwrite_output_dir \
    --dataloader_num_workers 0 \
    --weight_decay 0 \
    --lr_scheduler_type "constant" \
    --save_strategy "epoch" \
    --optim adamw_torch
```


Here is an example code to evaluate the out-of-domain performance on Lotte-Tech.
```bash
data_dir="./data/datasets/lotte/technology/test"
backbone_name_or_path="./data/dense-mlm/english-marco/baseline_train/distil"
output_dir="$backbone_name_or_path/evaluate/lotte/technology/test"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.evaluate.run_eval \
    --backbone_name_or_path $backbone_name_or_path \
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
    --topk 1000
```

Following the above commands, we train and evaluate a baseline model.
The evaluation results are
```python
{'NDCG@1': 0.22006, 'NDCG@3': 0.18594, 'NDCG@5': 0.17411, 'NDCG@10': 0.17425, 'NDCG@100': 0.23766}
{'MAP@1': 0.04028, 'MAP@3': 0.06987, 'MAP@5': 0.08294, 'MAP@10': 0.09638, 'MAP@100': 0.11561}
{'Recall@10': 0.16897, 'Recall@50': 0.30296, 'Recall@100': 0.3691, 'Recall@200': 0.44096, 'Recall@500': 0.53007, 'Recall@1000': 0.59645}
{'P@1': 0.22006, 'P@3': 0.16816, 'P@5': 0.14002, 'P@10': 0.1017, 'P@100': 0.02408}
{'MRR@10': 0.31285, 'MRR@100': 0.32316}
```
The results slightly differ from the reported results in our paper, largely due to different environments.