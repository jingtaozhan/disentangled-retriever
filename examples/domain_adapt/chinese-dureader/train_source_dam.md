# Training a Source DAM 

The source DAM aims to capture the corpus features in the source domain. Therefore, the REM module trained in this source domain can be generic because it will not be dependent on the source-domain features. 

```bash
output_dir="./data/adapt-mlm/chinese-dureader/train_rem/dam"

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

The trained model is actually our uploaded `jingtao/DAM-bert_base-mlm-dureader`. 
We freeze the trained source DAM and train REMs. The following links show how we train REMs for different ranking methods. Note that they all share the same source DAM module during training. 
- [Dense Retrieval](../../dense/chinese-dureader/train_rem.md)
- [uniCOIL](../../unicoil/chinese-dureader/train_rem.md)
- [SPLADE](../../splade/chinese-dureader/train_rem.md)
- [ColBERT](../../colbert/chinese-dureader/train_rem.md)
- [BERT re-ranker](../../rerank/chinese-dureader/train_rem.md)