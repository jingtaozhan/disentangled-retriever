## Unsupervisedly train DAM 

The source DAM aims to capture the corpus features in the source domain. Therefore, the REM module trained in this source domain can be generic because it will not be dependent on the source-domain features. 

```bash
output_dir="./data/adapt-mlm/english-marco/train_rem/dam"

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

The trained model is actually our uploaded `jingtao/DAM-bert_base-mlm-dureader`. 
We freeze the trained source DAM and train REMs. The following links show how we train REMs for different ranking methods. Note that they all share the same source DAM module during training. 
- [Dense Retrieval](../../dense/english-marco/train_rem.md)
- [uniCOIL](../../unicoil/english-marco/train_rem.md)
- [SPLADE](../../splade/english-marco/train_rem.md)
- [ColBERT](../../colbert/english-marco/train_rem.md)
- [BERT re-ranker](../../rerank/english-marco/train_rem.md)