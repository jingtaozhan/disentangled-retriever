# Train Dense Retrieval on Dureader

## Contrastive Finetuning

Here is the training command. We find model performance convergences after one epoch, and here we set the maximum training epoch to 1. The full batch size is $4 \times 32 = 128$. We use in-batch negatives because the dataset contains too many false negatives. `neg_per_query' is set to 0 accordingly.  
```bash
output_dir="./data/dense-mlm/chinese-dureader/baseline_train/contrast"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.finetune.run_contrast \
    --pooling average \
    --similarity_metric cos \
    --qrel_path ./data/datasets/dureader/qrels.train \
    --query_path ./data/datasets/dureader/query.train \
    --corpus_path ./data/datasets/dureader/corpus.tsv \
    --output_dir $output_dir \
    --model_name_or_path bert-base-chinese \
    --logging_steps 10 \
    --max_query_len 24 \
    --max_doc_len 384 \
    --per_device_train_batch_size 32 \
    --inv_temperature 20 \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --neg_per_query 0 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --dataloader_drop_last \
    --overwrite_output_dir \
    --dataloader_num_workers 0 \
    --weight_decay 0 \
    --lr_scheduler_type "constant" \
    --save_strategy "epoch" \
    --optim adamw_torch
```

Here is an example code to evaluate the out-of-domain performance on CPR-Ecommerce.
```bash
data_dir="./data/datasets/cpr-ecom"
backbone_name_or_path="./data/dense-mlm/chinese-dureader/baseline_train/contrast"
output_dir="$backbone_name_or_path/evaluate/cpr-ecom"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.evaluate.run_eval \
    --backbone_name_or_path $backbone_name_or_path \
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
    --topk 1000
```

Following the above commands, we train and evaluate a baseline model.
The evaluation results are
```python
{'NDCG@1': 0.167, 'NDCG@3': 0.22096, 'NDCG@5': 0.2439, 'NDCG@10': 0.26266, 'NDCG@100': 0.31282}
{'MAP@1': 0.167, 'MAP@3': 0.2075, 'MAP@5': 0.22015, 'MAP@10': 0.2279, 'MAP@100': 0.23689}
{'Recall@10': 0.374, 'Recall@50': 0.548, 'Recall@100': 0.623, 'Recall@200': 0.709, 'Recall@500': 0.789, 'Recall@1000': 0.842}
{'P@1': 0.167, 'P@3': 0.08667, 'P@5': 0.0632, 'P@10': 0.0374, 'P@100': 0.00623}
{'MRR@10': 0.2279, 'MRR@100': 0.23689}
```
The results slightly differ from the reported results in our paper, largely due to different environments.
