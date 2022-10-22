# Unsupervised Domain Adaption

To adapt to a new domain, just train a new DAM on the target corpus in an unsupervised manner. Here, we take CPR-Ecommerce as an example.

```bash
output_dir="./data/adapt-mlm/chinese-dureader/adapt_domain/cpr-ecom"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.adapt.run_adapt_with_mlm \
    --corpus_path ./data/datasets/cpr-ecom/corpus.tsv \
    --output_dir $output_dir \
    --model_name_or_path jingtao/DAM-bert_base-mlm-dureader \
    --logging_first_step \
    --logging_steps 50 \
    --max_seq_length 100 \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --warmup_steps 1000 \
    --fp16 \
    --learning_rate 2e-5 \
    --max_steps 20000 \
    --dataloader_drop_last \
    --overwrite_output_dir \
    --dataloader_num_workers 8 \
    --weight_decay 0.01 \
    --lr_scheduler_type "constant_with_warmup" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --optim adamw_torch 
```

Note that we use the source DAM `jingtao/DAM-bert_base-mlm-dureader` for initialization (see our paper for explanation) and train for 20,000 steps.
After this, you can use the trained DAM and combine it with any REM. The combination will be a very effective ranking model on Lotte-Technology. 

How to combine? Use the new DAM as the `backbone_name_or_path' argument. See the following instructions about different ranking methods:
- [Dense Retrieval](../../dense/chinese-dureader/inference.md)
- [uniCOIL](../../unicoil/chinese-dureader/inference.md)
- [SPLADE](../../splade/chinese-dureader/inference.md)
- [ColBERT](../../colbert/chinese-dureader/inference.md)
- [BERT re-ranker](../../rerank/chinese-dureader/inference.md)