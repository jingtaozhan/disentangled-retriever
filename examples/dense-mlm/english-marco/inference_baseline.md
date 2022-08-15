# Dense Retrieval Baseline 

Note: Different environments, e.g., different GPUs or different systems, may result in **slightly** different metric numbers. 

## Self-implemented Contrastively Trained Baseline

This repo is able to train a Dense Retrieval model that follows exactly the same contrastive training setting as Disentangled Dense Retrieval model. Therefore, we can rule out the influence of finetuning details when comparing models.

### TREC-Covid

```bash
data_dir="./data/datasets/trec-covid"
backbone_name_or_path="jingtao/Dense-bert_base-contrast-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/selfdr-contrast/trec-covid"

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
    --similarity_metric ip \
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.77, 'NDCG@3': 0.73123, 'NDCG@5': 0.70155, 'NDCG@10': 0.6475, 'NDCG@100': 0.42703}
{'MAP@1': 0.0022, 'MAP@3': 0.00564, 'MAP@5': 0.00821, 'MAP@10': 0.0138, 'MAP@100': 0.06319}
{'Recall@10': 0.01584, 'Recall@50': 0.05658, 'Recall@100': 0.0933, 'Recall@200': 0.14615, 'Recall@500': 0.23542, 'Recall@1000': 0.31436}
{'P@1': 0.84, 'P@3': 0.77333, 'P@5': 0.728, 'P@10': 0.672, 'P@100': 0.425}
{'MRR@10': 0.889, 'MRR@100': 0.88998}
```

### Lotte-Writing

### Lotte-Recreation

### Lotte-Technology

### Lotte-Lifestyle

### Lotte-Science


## Self-implemented Distilled Baseline

This repo is able to train a Dense Retrieval model that follows exactly the same distillation setting as Disentangled Dense Retrieval model. Therefore, we can rule out the influence of finetuning details when comparing models.

### TREC-Covid

```bash
data_dir="./data/datasets/trec-covid"
backbone_name_or_path="jingtao/Dense-bert_base-distil-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/selfdr-distil/trec-covid"

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
    --similarity_metric ip \
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.69, 'NDCG@3': 0.73123, 'NDCG@5': 0.73497, 'NDCG@10': 0.67803, 'NDCG@100': 0.47324}
{'MAP@1': 0.00198, 'MAP@3': 0.0056, 'MAP@5': 0.00909, 'MAP@10': 0.01555, 'MAP@100': 0.07537}
{'Recall@10': 0.01781, 'Recall@50': 0.06368, 'Recall@100': 0.10796, 'Recall@200': 0.16607, 'Recall@500': 0.26489, 'Recall@1000': 0.35476}
{'P@1': 0.8, 'P@3': 0.81333, 'P@5': 0.808, 'P@10': 0.726, 'P@100': 0.4792}
{'MRR@10': 0.8825, 'MRR@100': 0.8825}
```

### Lotte-Writing

### Lotte-Recreation

### Lotte-Technology

### Lotte-Lifestyle

### Lotte-Science

## TAS-Balanced Model

We also employ strong Dense Retrieval models in the literature as baselines. Here, we use TAS-Balanced (TAS-B) as an example. It collects the output scores of several cross-encoders, and then distils a Dense Retrieval model with these soft scores. 
In the following, we provide commands to evaluate TAS-B. 

>  Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang,Jimmy Lin,and Allan Hanbury. 2021. Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’21). 113–122.

### TREC-Covid

```bash
data_dir="./data/datasets/trec-covid"
backbone_name_or_path="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/tas-b/trec-covid"

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
    --pooling cls \
    --similarity_metric ip \
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.66, 'NDCG@3': 0.65838, 'NDCG@5': 0.62073, 'NDCG@10': 0.57061, 'NDCG@100': 0.41043}
{'MAP@1': 0.00189, 'MAP@3': 0.00507, 'MAP@5': 0.00743, 'MAP@10': 0.01217, 'MAP@100': 0.06064}
{'Recall@10': 0.014, 'Recall@50': 0.05514, 'Recall@100': 0.0926, 'Recall@200': 0.14472, 'Recall@500': 0.24482, 'Recall@1000': 0.33287}
{'P@1': 0.74, 'P@3': 0.72, 'P@5': 0.668, 'P@10': 0.6, 'P@100': 0.418}
{'MRR@10': 0.808, 'MRR@100': 0.81085}
```


### Lotte Writing

```bash
data_dir="./data/datasets/lotte/writing/test"
backbone_name_or_path="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/tas-b/lotte/writing/test"

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
    --pooling cls \
    --similarity_metric ip \
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.387, 'NDCG@3': 0.34571, 'NDCG@5': 0.33658, 'NDCG@10': 0.34618, 'NDCG@100': 0.41402}
{'MAP@1': 0.09221, 'MAP@3': 0.1766, 'MAP@5': 0.21312, 'MAP@10': 0.24601, 'MAP@100': 0.2733}
{'Recall@10': 0.35785, 'Recall@50': 0.51282, 'Recall@100': 0.56416, 'Recall@200': 0.61236, 'Recall@500': 0.67072, 'Recall@1000': 0.71252}
{'P@1': 0.387, 'P@3': 0.31133, 'P@5': 0.2582, 'P@10': 0.17715, 'P@100': 0.03042}
{'MRR@10': 0.49137, 'MRR@100': 0.49842}
```

### Lotte Recreation

```bash
data_dir="./data/datasets/lotte/recreation/test"
backbone_name_or_path="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/tas-b/lotte/recreation/test"

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
    --pooling cls \
    --similarity_metric ip \
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.41109, 'NDCG@3': 0.36251, 'NDCG@5': 0.3665, 'NDCG@10': 0.39033, 'NDCG@100': 0.45452}
{'MAP@1': 0.16758, 'MAP@3': 0.25759, 'MAP@5': 0.28554, 'MAP@10': 0.30746, 'MAP@100': 0.32661}
{'Recall@10': 0.43616, 'Recall@50': 0.59189, 'Recall@100': 0.65136, 'Recall@200': 0.70302, 'Recall@500': 0.77229, 'Recall@1000': 0.8108}
{'P@1': 0.41109, 'P@3': 0.27406, 'P@5': 0.20699, 'P@10': 0.13132, 'P@100': 0.0211}
{'MRR@10': 0.49997, 'MRR@100': 0.50717}
```

### Lotte Technology

```bash
data_dir="./data/datasets/lotte/technology/test"
backbone_name_or_path="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/tas-b/lotte/technology/test"

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
    --pooling cls \
    --similarity_metric ip \
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.20309, 'NDCG@3': 0.17887, 'NDCG@5': 0.16711, 'NDCG@10': 0.16772, 'NDCG@100': 0.23029}
{'MAP@1': 0.03824, 'MAP@3': 0.06853, 'MAP@5': 0.08073, 'MAP@10': 0.09333, 'MAP@100': 0.11184}
{'Recall@10': 0.16626, 'Recall@50': 0.2959, 'Recall@100': 0.36157, 'Recall@200': 0.43306, 'Recall@500': 0.52957, 'Recall@1000': 0.59454}
{'P@1': 0.20309, 'P@3': 0.16334, 'P@5': 0.13403, 'P@10': 0.09726, 'P@100': 0.02352}
{'MRR@10': 0.29441, 'MRR@100': 0.30483}
```

### Lotte Lifestyle

```bash
data_dir="./data/datasets/lotte/lifestyle/test"
backbone_name_or_path="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/tas-b/lotte/lifestyle/test"

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
    --pooling cls \
    --similarity_metric ip \
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.45804, 'NDCG@3': 0.40863, 'NDCG@5': 0.39389, 'NDCG@10': 0.40576, 'NDCG@100': 0.49389}
{'MAP@1': 0.13287, 'MAP@3': 0.2301, 'MAP@5': 0.2673, 'MAP@10': 0.29941, 'MAP@100': 0.33277}
{'Recall@10': 0.42122, 'Recall@50': 0.61766, 'Recall@100': 0.68813, 'Recall@200': 0.75138, 'Recall@500': 0.82392, 'Recall@1000': 0.8725}
{'P@1': 0.45804, 'P@3': 0.35065, 'P@5': 0.28062, 'P@10': 0.18846, 'P@100': 0.03347}
{'MRR@10': 0.566, 'MRR@100': 0.57222}
```

### Lotte Science

```bash
data_dir="./data/datasets/lotte/science/test"
backbone_name_or_path="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/tas-b/lotte/science/test"

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
    --pooling cls \
    --similarity_metric ip \
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.16014, 'NDCG@3': 0.13619, 'NDCG@5': 0.12951, 'NDCG@10': 0.13299, 'NDCG@100': 0.17988}
{'MAP@1': 0.03136, 'MAP@3': 0.05529, 'MAP@5': 0.06552, 'MAP@10': 0.07737, 'MAP@100': 0.09137}
{'Recall@10': 0.1313, 'Recall@50': 0.23231, 'Recall@100': 0.28147, 'Recall@200': 0.3353, 'Recall@500': 0.40985, 'Recall@1000': 0.46392}
{'P@1': 0.16014, 'P@3': 0.12279, 'P@5': 0.10302, 'P@10': 0.07794, 'P@100': 0.01823}
{'MRR@10': 0.22654, 'MRR@100': 0.23567}
```