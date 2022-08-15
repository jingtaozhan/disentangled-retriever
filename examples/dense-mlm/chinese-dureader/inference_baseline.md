# Dense Retrieval Baseline 

Note: Different environments, e.g., different GPUs or different systems, may result in **slightly** different metric numbers. 

## Self-implemented Contrastively Trained Baseline

This repo is able to train a Dense Retrieval model that follows exactly the same contrastive training setting as Disentangled Dense Retrieval model. Therefore, we can rule out the influence of finetuning details when comparing models.

### CPR-Ecommerce

```bash
data_dir="./data/datasets/cpr-ecom"
backbone_name_or_path="jingtao/Dense-bert_base-contrast-dureader"
output_dir="./data/dense-mlm/chinese-dureader/inference_baseline/selfdr-contrast/cpr-ecom"

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
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.167, 'NDCG@3': 0.22196, 'NDCG@5': 0.24365, 'NDCG@10': 0.26278, 'NDCG@100': 0.31312}
{'MAP@1': 0.167, 'MAP@3': 0.20817, 'MAP@5': 0.22012, 'MAP@10': 0.22805, 'MAP@100': 0.23707}
{'Recall@10': 0.374, 'Recall@50': 0.548, 'Recall@100': 0.624, 'Recall@200': 0.708, 'Recall@500': 0.789, 'Recall@1000': 0.844}
{'P@1': 0.167, 'P@3': 0.08733, 'P@5': 0.063, 'P@10': 0.0374, 'P@100': 0.00624}
{'MRR@10': 0.22805, 'MRR@100': 0.23707}
```

### CPR-Video

```bash
data_dir="./data/datasets/cpr-video"
backbone_name_or_path="jingtao/Dense-bert_base-contrast-dureader"
output_dir="./data/dense-mlm/chinese-dureader/inference_baseline/selfdr-contrast/cpr-video"

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
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.147, 'NDCG@3': 0.20775, 'NDCG@5': 0.22544, 'NDCG@10': 0.25091, 'NDCG@100': 0.30581}
{'MAP@1': 0.147, 'MAP@3': 0.1925, 'MAP@5': 0.2023, 'MAP@10': 0.21296, 'MAP@100': 0.22341}
{'Recall@10': 0.373, 'Recall@50': 0.564, 'Recall@100': 0.639, 'Recall@200': 0.702, 'Recall@500': 0.78, 'Recall@1000': 0.825}
{'P@1': 0.147, 'P@3': 0.084, 'P@5': 0.059, 'P@10': 0.0373, 'P@100': 0.00639}
{'MRR@10': 0.21296, 'MRR@100': 0.22341}
```

### CPR-Medical

```bash
data_dir="./data/datasets/cpr-medical"
backbone_name_or_path="jingtao/Dense-bert_base-contrast-dureader"
output_dir="./data/dense-mlm/chinese-dureader/inference_baseline/selfdr-contrast/cpr-medical"

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
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
```

### cMedQAv2

```bash
data_dir="./data/datasets/cmedqav2"
backbone_name_or_path="jingtao/Dense-bert_base-contrast-dureader"
output_dir="./data/dense-mlm/chinese-dureader/inference_baseline/selfdr-contrast/cmedqav2"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.evaluate.run_eval \
    --backbone_name_or_path $backbone_name_or_path \
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
{'NDCG@1': 0.10725, 'NDCG@3': 0.09907, 'NDCG@5': 0.10205, 'NDCG@10': 0.11039, 'NDCG@100': 0.1396}
{'MAP@1': 0.06734, 'MAP@3': 0.08156, 'MAP@5': 0.08544, 'MAP@10': 0.08932, 'MAP@100': 0.09465}
{'Recall@10': 0.13274, 'Recall@50': 0.20917, 'Recall@100': 0.26152, 'Recall@200': 0.32005, 'Recall@500': 0.4191, 'Recall@1000': 0.49853}
{'P@1': 0.10725, 'P@3': 0.0545, 'P@5': 0.0384, 'P@10': 0.02412, 'P@100': 0.00484}
{'MRR@10': 0.133, 'MRR@100': 0.13832}
```

## Self-implemented Distilled Baseline

This repo is able to train a Dense Retrieval model that follows exactly the same distillation setting as Disentangled Dense Retrieval model. Therefore, we can rule out the influence of finetuning details when comparing models.

### CPR-Ecommerce

```bash
data_dir="./data/datasets/cpr-ecom"
backbone_name_or_path="jingtao/Dense-bert_base-distil-dureader"
output_dir="./data/dense-mlm/chinese-dureader/inference_baseline/selfdr-distil/cpr-ecom"

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
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.189, 'NDCG@3': 0.24359, 'NDCG@5': 0.26959, 'NDCG@10': 0.28715, 'NDCG@100': 0.33692}
{'MAP@1': 0.189, 'MAP@3': 0.23, 'MAP@5': 0.24445, 'MAP@10': 0.25176, 'MAP@100': 0.26072}
{'Recall@10': 0.4, 'Recall@50': 0.57, 'Recall@100': 0.647, 'Recall@200': 0.718, 'Recall@500': 0.779, 'Recall@1000': 0.824}
{'P@1': 0.189, 'P@3': 0.09433, 'P@5': 0.0692, 'P@10': 0.04, 'P@100': 0.00647}
{'MRR@10': 0.25176, 'MRR@100': 0.26072}
```

### CPR-Video

```bash
data_dir="./data/datasets/cpr-video"
backbone_name_or_path="jingtao/Dense-bert_base-distil-dureader"
output_dir="./data/dense-mlm/chinese-dureader/inference_baseline/selfdr-distil/cpr-video"

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
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.168, 'NDCG@3': 0.22651, 'NDCG@5': 0.25323, 'NDCG@10': 0.28384, 'NDCG@100': 0.34197}
{'MAP@1': 0.168, 'MAP@3': 0.21217, 'MAP@5': 0.22697, 'MAP@10': 0.23952, 'MAP@100': 0.25087}
{'Recall@10': 0.428, 'Recall@50': 0.639, 'Recall@100': 0.706, 'Recall@200': 0.756, 'Recall@500': 0.827, 'Recall@1000': 0.858}
{'P@1': 0.168, 'P@3': 0.08933, 'P@5': 0.0666, 'P@10': 0.0428, 'P@100': 0.00706}
{'MRR@10': 0.23952, 'MRR@100': 0.25087}
```

### CPR-Medical

```bash
data_dir="./data/datasets/cpr-medical"
backbone_name_or_path="jingtao/Dense-bert_base-distil-dureader"
output_dir="./data/dense-mlm/chinese-dureader/inference_baseline/selfdr-distil/cpr-medical"

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
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python

```

### cMedQAv2

```bash
data_dir="./data/datasets/cmedqav2"
backbone_name_or_path="jingtao/Dense-bert_base-distil-dureader"
output_dir="./data/dense-mlm/chinese-dureader/inference_baseline/selfdr-distil/cmedqav2"

python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.evaluate.run_eval \
    --backbone_name_or_path $backbone_name_or_path \
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
{'NDCG@1': 0.10525, 'NDCG@3': 0.09631, 'NDCG@5': 0.09942, 'NDCG@10': 0.10668, 'NDCG@100': 0.13411}
{'MAP@1': 0.06402, 'MAP@3': 0.0786, 'MAP@5': 0.08242, 'MAP@10': 0.08585, 'MAP@100': 0.09086}
{'Recall@10': 0.12841, 'Recall@50': 0.20241, 'Recall@100': 0.24852, 'Recall@200': 0.30292, 'Recall@500': 0.39167, 'Recall@1000': 0.46816}
{'P@1': 0.10525, 'P@3': 0.05342, 'P@5': 0.03765, 'P@10': 0.0231, 'P@100': 0.00463}
{'MRR@10': 0.13055, 'MRR@100': 0.13538}
```