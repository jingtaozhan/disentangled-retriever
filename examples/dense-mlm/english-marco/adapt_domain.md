# Unsupervised Domain Adaption

We use Lotte-Technology as an example. To adapt to an unseen domain, we unsupervisedly train a separate DAM model based on the corpus.
```bash
output_dir="./data/dense-mlm/english-marco/adapt_domain/lotte/technology/test"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.adapt.run_adapt_with_mlm \
    --corpus_path ./data/datasets/lotte/technology/test/corpus.tsv \
    --output_dir $output_dir \
    --model_name_or_path jingtao/DAM-bert_base-mlm-msmarco \
    --logging_first_step \
    --logging_steps 50 \
    --max_seq_length 190 \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --warmup_steps 1000 \
    --fp16 \
    --learning_rate 2e-5 \
    --max_steps 50000 \
    --dataloader_drop_last \
    --overwrite_output_dir \
    --dataloader_num_workers 8 \
    --weight_decay 0.01 \
    --lr_scheduler_type "constant_with_warmup" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --optim adamw_torch 
```

Now we assemble the trained DAM and the REM model. We use the contrastively finetuned REM as an example. 
```bash
data_dir="./data/datasets/lotte/technology/test"
backbone_name_or_path="./data/dense-mlm/english-marco/adapt_domain/lotte/technology/test"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-contrast-msmarco/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/english-marco/adapt_domain/lotte/technology/test/evaluate/contrast"

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

The evaluation results are
```bash
{'NDCG@1': 0.26697, 'NDCG@3': 0.22438, 'NDCG@5': 0.2099, 'NDCG@10': 0.20929, 'NDCG@100': 0.28923}
{'MAP@1': 0.04649, 'MAP@3': 0.08394, 'MAP@5': 0.10064, 'MAP@10': 0.11739, 'MAP@100': 0.14307}
{'Recall@10': 0.20156, 'Recall@50': 0.37467, 'Recall@100': 0.45131, 'Recall@200': 0.53181, 'Recall@500': 0.63878, 'Recall@1000': 0.71411}
{'P@1': 0.26697, 'P@3': 0.20459, 'P@5': 0.16996, 'P@10': 0.12345, 'P@100': 0.02983}
{'MRR@10': 0.36895, 'MRR@100': 0.37979}
```
If the distilled REM is used, the evaluation results are
```python
{'NDCG@1': 0.25649, 'NDCG@3': 0.21855, 'NDCG@5': 0.20464, 'NDCG@10': 0.20648, 'NDCG@100': 0.28399}
{'MAP@1': 0.04581, 'MAP@3': 0.08228, 'MAP@5': 0.09746, 'MAP@10': 0.11558, 'MAP@100': 0.14035}
{'Recall@10': 0.20276, 'Recall@50': 0.3655, 'Recall@100': 0.44248, 'Recall@200': 0.52052, 'Recall@500': 0.63083, 'Recall@1000': 0.70175}
{'P@1': 0.25649, 'P@3': 0.20043, 'P@5': 0.16587, 'P@10': 0.12181, 'P@100': 0.02921}
{'MRR@10': 0.36055, 'MRR@100': 0.37163}
```
The results slightly differ from the reported results in our paper, largely due to different environments.
