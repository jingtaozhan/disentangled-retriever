# Inference with Trained Models

Note: Different environments, e.g., different GPUs or different systems, may result in **slightly** different metric numbers. 

## CPR-Ecommerce

<details>
<summary>Inference with unsupervised DAM and contrastively trained REM.</summary>

```bash
data_dir="./data/datasets/cpr-ecom"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-dureader-cpr_ecom"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-contrast-dureader/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/chinese-dureader/inference/cpr-ecom/contrast"

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
    --save_corpus_embed \
    --save_query_embed \
    --merge_lora
```

The output is:
```python
{'NDCG@1': 0.2, 'NDCG@3': 0.26958, 'NDCG@5': 0.29015, 'NDCG@10': 0.31295, 'NDCG@100': 0.36578}
{'MAP@1': 0.2, 'MAP@3': 0.2525, 'MAP@5': 0.2639, 'MAP@10': 0.27321, 'MAP@100': 0.28265}
{'Recall@10': 0.44, 'Recall@50': 0.618, 'Recall@100': 0.703, 'Recall@200': 0.764, 'Recall@500': 0.832, 'Recall@1000': 0.882}
{'P@1': 0.2, 'P@3': 0.10633, 'P@5': 0.0738, 'P@10': 0.044, 'P@100': 0.00703}
{'MRR@10': 0.27321, 'MRR@100': 0.28265}
```

</details>


<details>
<summary>Inference with unsupervised DAM and distilled REM.</summary>
  
```bash
data_dir="./data/datasets/cpr-ecom"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-dureader-cpr_ecom"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-distil-dureader/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/chinese-dureader/inference/cpr-ecom/distil"

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
    --save_corpus_embed \
    --save_query_embed \
    --merge_lora
```

The output is:
```python
{'NDCG@1': 0.207, 'NDCG@3': 0.27182, 'NDCG@5': 0.29454, 'NDCG@10': 0.31769, 'NDCG@100': 0.36599}
{'MAP@1': 0.207, 'MAP@3': 0.25617, 'MAP@5': 0.26882, 'MAP@10': 0.27829, 'MAP@100': 0.28715}
{'Recall@10': 0.444, 'Recall@50': 0.612, 'Recall@100': 0.682, 'Recall@200': 0.755, 'Recall@500': 0.822, 'Recall@1000': 0.861}
{'P@1': 0.207, 'P@3': 0.10567, 'P@5': 0.0744, 'P@10': 0.0444, 'P@100': 0.00682}
{'MRR@10': 0.27829, 'MRR@100': 0.28715}
```
</details>

## CPR-Video

<details>
<summary>Inference with unsupervised DAM and contrastively trained REM.</summary>

```bash
data_dir="./data/datasets/cpr-video"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-dureader-cpr_video"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-contrast-dureader/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/chinese-dureader/inference/cpr-video/contrast"

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
    --save_corpus_embed \
    --save_query_embed \
    --merge_lora
```

The output is:
```python
{'NDCG@1': 0.154, 'NDCG@3': 0.22421, 'NDCG@5': 0.24211, 'NDCG@10': 0.26897, 'NDCG@100': 0.32749}
{'MAP@1': 0.154, 'MAP@3': 0.207, 'MAP@5': 0.2168, 'MAP@10': 0.22808, 'MAP@100': 0.23919}
{'Recall@10': 0.4, 'Recall@50': 0.61, 'Recall@100': 0.684, 'Recall@200': 0.746, 'Recall@500': 0.809, 'Recall@1000': 0.859}
{'P@1': 0.154, 'P@3': 0.09133, 'P@5': 0.0636, 'P@10': 0.04, 'P@100': 0.00684}
{'MRR@10': 0.22808, 'MRR@100': 0.23919}
```
</details>

<details>
<summary>Inference with unsupervised DAM and distilled REM.</summary>

```bash
data_dir="./data/datasets/cpr-video"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-dureader-cpr_video"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-distil-dureader/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/chinese-dureader/inference/cpr-video/distil"

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
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.172, 'NDCG@3': 0.24406, 'NDCG@5': 0.26846, 'NDCG@10': 0.29765, 'NDCG@100': 0.35751}
{'MAP@1': 0.172, 'MAP@3': 0.22583, 'MAP@5': 0.23943, 'MAP@10': 0.25153, 'MAP@100': 0.26286}
{'Recall@10': 0.446, 'Recall@50': 0.645, 'Recall@100': 0.737, 'Recall@200': 0.796, 'Recall@500': 0.845, 'Recall@1000': 0.887}
{'P@1': 0.172, 'P@3': 0.099, 'P@5': 0.0712, 'P@10': 0.0446, 'P@100': 0.00737}
{'MRR@10': 0.25153, 'MRR@100': 0.26286}
```

</details>

## CPR-Medical

<details>
<summary>Inference with unsupervised DAM and contrastively trained REM.</summary>

```bash
data_dir="./data/datasets/cpr-medical"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-dureader-cpr_medical"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-contrast-dureader/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/chinese-dureader/inference/cpr-medical/contrast"

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
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.25, 'NDCG@3': 0.30049, 'NDCG@5': 0.31331, 'NDCG@10': 0.32219, 'NDCG@100': 0.3437}
{'MAP@1': 0.25, 'MAP@3': 0.2885, 'MAP@5': 0.29565, 'MAP@10': 0.29922, 'MAP@100': 0.30302}
{'Recall@10': 0.394, 'Recall@50': 0.465, 'Recall@100': 0.502, 'Recall@200': 0.54, 'Recall@500': 0.619, 'Recall@1000': 0.668}
{'P@1': 0.25, 'P@3': 0.11167, 'P@5': 0.0732, 'P@10': 0.0394, 'P@100': 0.00502}
{'MRR@10': 0.29922, 'MRR@100': 0.30302}
```
</details>

<details>
<summary>Inference with unsupervised DAM and distilled trained REM.</summary>

```bash
data_dir="./data/datasets/cpr-medical"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-dureader-cpr_medical"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-distil-dureader/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/chinese-dureader/inference/cpr-medical/distil"

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
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.235, 'NDCG@3': 0.28009, 'NDCG@5': 0.29115, 'NDCG@10': 0.30164, 'NDCG@100': 0.3257}
{'MAP@1': 0.235, 'MAP@3': 0.26967, 'MAP@5': 0.27577, 'MAP@10': 0.27999, 'MAP@100': 0.28447}
{'Recall@10': 0.37, 'Recall@50': 0.45, 'Recall@100': 0.488, 'Recall@200': 0.531, 'Recall@500': 0.59, 'Recall@1000': 0.643}
{'P@1': 0.235, 'P@3': 0.10333, 'P@5': 0.0674, 'P@10': 0.037, 'P@100': 0.00488}
{'MRR@10': 0.27999, 'MRR@100': 0.28447}
```
</details>

## cMedQAv2

<details>
<summary>Inference with unsupervised DAM and contrastively trained REM.</summary>

```bash
data_dir="./data/datasets/cmedqav2"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-dureader-cmedqav2"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-contrast-dureader/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/chinese-dureader/inference/cmedqav2/contrast"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.evaluate.run_eval \
    --backbone_name_or_path $backbone_name_or_path \
    --adapter_name_or_path $adapter_name_or_path \
    --corpus_path $data_dir/corpus.tsv \
    --query_path $data_dir/query.test \
    --qrel_path $data_dir/qrels.test \
    --output_dir $output_dir \
    --out_corpus_dir $output_dir/corpus \
    --out_query_dir $output_dir/test \
    --per_device_eval_batch_size 48 \
    --overwrite_output_dir \
    --dataloader_num_workers 8 \
    --pooling average \
    --similarity_metric cos \
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.10675, 'NDCG@3': 0.09839, 'NDCG@5': 0.10381, 'NDCG@10': 0.11426, 'NDCG@100': 0.14572}
{'MAP@1': 0.06571, 'MAP@3': 0.08077, 'MAP@5': 0.08574, 'MAP@10': 0.09071, 'MAP@100': 0.09651}
{'Recall@10': 0.14432, 'Recall@50': 0.22905, 'Recall@100': 0.28208, 'Recall@200': 0.34291, 'Recall@500': 0.44528, 'Recall@1000': 0.53191}
{'P@1': 0.10675, 'P@3': 0.05433, 'P@5': 0.0399, 'P@10': 0.02567, 'P@100': 0.00522}
{'MRR@10': 0.13458, 'MRR@100': 0.14025}
```
</details>


<details>
<summary>Inference with unsupervised DAM and distiled REM.</summary>

```bash
data_dir="./data/datasets/cmedqav2"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-dureader-cmedqav2"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-distil-dureader/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/chinese-dureader/inference/cmedqav2/distil"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.evaluate.run_eval \
    --backbone_name_or_path $backbone_name_or_path \
    --adapter_name_or_path $adapter_name_or_path \
    --corpus_path $data_dir/corpus.tsv \
    --query_path $data_dir/query.test \
    --qrel_path $data_dir/qrels.test \
    --output_dir $output_dir \
    --out_corpus_dir $output_dir/corpus \
    --out_query_dir $output_dir/test \
    --per_device_eval_batch_size 48 \
    --overwrite_output_dir \
    --dataloader_num_workers 8 \
    --pooling average \
    --similarity_metric cos \
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.10375, 'NDCG@3': 0.0964, 'NDCG@5': 0.10037, 'NDCG@10': 0.10811, 'NDCG@100': 0.13776}
{'MAP@1': 0.06357, 'MAP@3': 0.079, 'MAP@5': 0.08329, 'MAP@10': 0.08706, 'MAP@100': 0.09242}
{'Recall@10': 0.13129, 'Recall@50': 0.21023, 'Recall@100': 0.26219, 'Recall@200': 0.32153, 'Recall@500': 0.41557, 'Recall@1000': 0.49979}
{'P@1': 0.10375, 'P@3': 0.0535, 'P@5': 0.03835, 'P@10': 0.02372, 'P@100': 0.00482}
{'MRR@10': 0.12967, 'MRR@100': 0.13513}
```
</details>