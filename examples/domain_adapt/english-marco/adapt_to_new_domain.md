# Unsupervised Domain Adaption

To adapt to a new domain, just train a new DAM on the target corpus in an unsupervised manner. Here, we take Lotte-Technology as an example.

```bash
output_dir="./data/adapt-mlm/english-marco/adapt_domain/lotte/technology/test"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.adapt.run_adapt_with_mlm \
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

Note that we use the source DAM `jingtao/DAM-bert_base-mlm-msmarco` for initialization (see our paper for explanation) and train for 10,000 steps.
After this, you can use the trained DAM and combine it with any REM. The combination will be a very effective ranking model on Lotte-Technology. 

How to combine? Use the new DAM as the `backbone_name_or_path' argument. See the following instructions about different ranking methods:
- [Dense Retrieval](../../dense/english-marco/inference.md)
- [uniCOIL](../../unicoil/english-marco/inference.md)
- [SPLADE](../../splade/english-marco/inference.md)
- [ColBERT](../../colbert/english-marco/inference.md)
- [BERT re-ranker](../../rerank/english-marco/inference.md)