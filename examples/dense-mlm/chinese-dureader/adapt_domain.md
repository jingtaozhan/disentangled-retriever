# Unsupervised Domain Adaption

We use CPR-Ecommerce as an example. To adapt to an unseen domain, we unsupervisedly train a separate DAM model based on the corpus.
```bash
output_dir="./data/dense-mlm/chinese-dureader/adapt_domain/cpr-ecom"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.adapt.run_adapt_with_mlm \
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

Now we assemble the trained DAM and the REM model. We use the contrastively finetuned REM as an example. 
```bash
data_dir="./data/datasets/cpr-ecom"
backbone_name_or_path="./data/dense-mlm/chinese-dureader/adapt_domain/cpr-ecom"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-contrast-dureader/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/chinese-dureader/adapt_domain/cpr-ecom/evaluate/contrast"

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

The evaluation results are
```bash

```
