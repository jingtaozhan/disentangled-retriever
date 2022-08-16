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

```bash
data_dir="./data/datasets/lotte/writing/test"
backbone_name_or_path="jingtao/Dense-bert_base-contrast-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/selfdr-contrast/lotte/writing/test"

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
    --pooling average \
    --similarity_metric ip \
    --topk 1000
```

The output is:
```python
{'NDCG@1': 0.4255, 'NDCG@3': 0.35952, 'NDCG@5': 0.34422, 'NDCG@10': 0.3522, 'NDCG@100': 0.41409}
{'MAP@1': 0.10083, 'MAP@3': 0.1829, 'MAP@5': 0.21775, 'MAP@10': 0.24974, 'MAP@100': 0.27377}
{'Recall@10': 0.3546, 'Recall@50': 0.49345, 'Recall@100': 0.54952, 'Recall@200': 0.59704, 'Recall@500': 0.65965, 'Recall@1000': 0.70716}
{'P@1': 0.4255, 'P@3': 0.31917, 'P@5': 0.2592, 'P@10': 0.1775, 'P@100': 0.02928}
{'MRR@10': 0.51825, 'MRR@100': 0.52447}
```

### Lotte-Recreation

```bash
data_dir="./data/datasets/lotte/recreation/test"
backbone_name_or_path="jingtao/Dense-bert_base-contrast-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/selfdr-contrast/lotte/recreation/test"

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
    --pooling average \
    --similarity_metric ip \
    --topk 1000
```

The output is:
```python
{'NDCG@1': 0.38611, 'NDCG@3': 0.33902, 'NDCG@5': 0.34039, 'NDCG@10': 0.35985, 'NDCG@100': 0.41873}
{'MAP@1': 0.16066, 'MAP@3': 0.24234, 'MAP@5': 0.26534, 'MAP@10': 0.2839, 'MAP@100': 0.30083}
{'Recall@10': 0.39424, 'Recall@50': 0.5361, 'Recall@100': 0.59236, 'Recall@200': 0.64393, 'Recall@500': 0.71522, 'Recall@1000': 0.76853}
{'P@1': 0.38611, 'P@3': 0.25391, 'P@5': 0.18991, 'P@10': 0.11923, 'P@100': 0.01931}
{'MRR@10': 0.46592, 'MRR@100': 0.47325}
```

### Lotte-Technology

```bash
data_dir="./data/datasets/lotte/technology/test"
backbone_name_or_path="jingtao/Dense-bert_base-contrast-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/selfdr-contrast/lotte/technology/test"

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
    --pooling average \
    --similarity_metric ip \
    --topk 1000
```

The output is:
```python
{'NDCG@1': 0.17914, 'NDCG@3': 0.15434, 'NDCG@5': 0.14315, 'NDCG@10': 0.14098, 'NDCG@100': 0.1947}
{'MAP@1': 0.03221, 'MAP@3': 0.05599, 'MAP@5': 0.06513, 'MAP@10': 0.07525, 'MAP@100': 0.08977}
{'Recall@10': 0.13565, 'Recall@50': 0.24957, 'Recall@100': 0.30783, 'Recall@200': 0.37202, 'Recall@500': 0.4593, 'Recall@1000': 0.52878}
{'P@1': 0.17914, 'P@3': 0.14138, 'P@5': 0.11607, 'P@10': 0.08179, 'P@100': 0.01993}
{'MRR@10': 0.26424, 'MRR@100': 0.27456}
```

### Lotte-Lifestyle

```bash
data_dir="./data/datasets/lotte/lifestyle/test"
backbone_name_or_path="jingtao/Dense-bert_base-contrast-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/selfdr-contrast/lotte/lifestyle/test"

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
    --pooling average \
    --similarity_metric ip \
    --topk 1000
```

The output is:
```python
{'NDCG@1': 0.43556, 'NDCG@3': 0.37723, 'NDCG@5': 0.36501, 'NDCG@10': 0.37494, 'NDCG@100': 0.46105}
{'MAP@1': 0.12378, 'MAP@3': 0.21046, 'MAP@5': 0.24324, 'MAP@10': 0.27167, 'MAP@100': 0.30285}
{'Recall@10': 0.38986, 'Recall@50': 0.57783, 'Recall@100': 0.65061, 'Recall@200': 0.71531, 'Recall@500': 0.79266, 'Recall@1000': 0.84597}
{'P@1': 0.43556, 'P@3': 0.31968, 'P@5': 0.25654, 'P@10': 0.17118, 'P@100': 0.03137}
{'MRR@10': 0.53946, 'MRR@100': 0.54631}
```

### Lotte-Science

```bash
data_dir="./data/datasets/lotte/science/test"
backbone_name_or_path="jingtao/Dense-bert_base-contrast-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/selfdr-contrast/lotte/science/test"

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
    --pooling average \
    --similarity_metric ip \
    --topk 1000
```

The output is:
```python
{'NDCG@1': 0.15221, 'NDCG@3': 0.12736, 'NDCG@5': 0.12087, 'NDCG@10': 0.12089, 'NDCG@100': 0.15884}
{'MAP@1': 0.03107, 'MAP@3': 0.05197, 'MAP@5': 0.06105, 'MAP@10': 0.07047, 'MAP@100': 0.08106}
{'Recall@10': 0.11484, 'Recall@50': 0.19616, 'Recall@100': 0.23721, 'Recall@200': 0.28007, 'Recall@500': 0.34232, 'Recall@1000': 0.39754}
{'P@1': 0.15221, 'P@3': 0.1137, 'P@5': 0.09509, 'P@10': 0.06911, 'P@100': 0.01563}
{'MRR@10': 0.21364, 'MRR@100': 0.22212}
```

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

```bash
data_dir="./data/datasets/lotte/writing/test"
backbone_name_or_path="jingtao/Dense-bert_base-distil-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/selfdr-distil/lotte/writing/test"

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
    --pooling average \
    --similarity_metric ip \
    --topk 1000
```

The output is:
```python
{'NDCG@1': 0.462, 'NDCG@3': 0.40215, 'NDCG@5': 0.3851, 'NDCG@10': 0.39241, 'NDCG@100': 0.459}
{'MAP@1': 0.11087, 'MAP@3': 0.20842, 'MAP@5': 0.2496, 'MAP@10': 0.28574, 'MAP@100': 0.31399}
{'Recall@10': 0.39603, 'Recall@50': 0.55223, 'Recall@100': 0.59936, 'Recall@200': 0.64211, 'Recall@500': 0.69779, 'Recall@1000': 0.7338}
{'P@1': 0.462, 'P@3': 0.36117, 'P@5': 0.2922, 'P@10': 0.1962, 'P@100': 0.03201}
{'MRR@10': 0.5604, 'MRR@100': 0.5666}
```

### Lotte-Recreation

```bash
data_dir="./data/datasets/lotte/recreation/test"
backbone_name_or_path="jingtao/Dense-bert_base-distil-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/selfdr-distil/lotte/recreation/test"

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
    --pooling average \
    --similarity_metric ip \
    --topk 1000
```

The output is:
```python
{'NDCG@1': 0.41259, 'NDCG@3': 0.37043, 'NDCG@5': 0.37487, 'NDCG@10': 0.39874, 'NDCG@100': 0.46159}
{'MAP@1': 0.17535, 'MAP@3': 0.26608, 'MAP@5': 0.29397, 'MAP@10': 0.31642, 'MAP@100': 0.33586}
{'Recall@10': 0.44351, 'Recall@50': 0.59607, 'Recall@100': 0.65082, 'Recall@200': 0.70435, 'Recall@500': 0.75594, 'Recall@1000': 0.79749}
{'P@1': 0.41259, 'P@3': 0.27855, 'P@5': 0.21029, 'P@10': 0.13352, 'P@100': 0.02121}
{'MRR@10': 0.50322, 'MRR@100': 0.51024}
```

### Lotte-Technology

```bash
data_dir="./data/datasets/lotte/technology/test"
backbone_name_or_path="jingtao/Dense-bert_base-distil-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/selfdr-distil/lotte/technology/test"

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
    --pooling average \
    --similarity_metric ip \
    --topk 1000
```

The output is:
```python
{'NDCG@1': 0.21956, 'NDCG@3': 0.18469, 'NDCG@5': 0.17385, 'NDCG@10': 0.17548, 'NDCG@100': 0.23622}
{'MAP@1': 0.0404, 'MAP@3': 0.07033, 'MAP@5': 0.08327, 'MAP@10': 0.09748, 'MAP@100': 0.11609}
{'Recall@10': 0.17222, 'Recall@50': 0.29934, 'Recall@100': 0.36435, 'Recall@200': 0.42824, 'Recall@500': 0.52112, 'Recall@1000': 0.58595}
{'P@1': 0.21956, 'P@3': 0.1665, 'P@5': 0.13962, 'P@10': 0.10304, 'P@100': 0.02391}
{'MRR@10': 0.30952, 'MRR@100': 0.3195}
```

### Lotte-Lifestyle

```bash
data_dir="./data/datasets/lotte/lifestyle/test"
backbone_name_or_path="jingtao/Dense-bert_base-distil-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/selfdr-distil/lotte/lifestyle/test"

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
    --pooling average \
    --similarity_metric ip \
    --topk 1000
```

The output is:
```python
{'NDCG@1': 0.47552, 'NDCG@3': 0.41454, 'NDCG@5': 0.39704, 'NDCG@10': 0.40897, 'NDCG@100': 0.49467}
{'MAP@1': 0.13378, 'MAP@3': 0.23258, 'MAP@5': 0.2699, 'MAP@10': 0.30296, 'MAP@100': 0.33521}
{'Recall@10': 0.42452, 'Recall@50': 0.61207, 'Recall@100': 0.68221, 'Recall@200': 0.74736, 'Recall@500': 0.82103, 'Recall@1000': 0.86609}
{'P@1': 0.47552, 'P@3': 0.35348, 'P@5': 0.28032, 'P@10': 0.18781, 'P@100': 0.03303}
{'MRR@10': 0.57378, 'MRR@100': 0.58053}
```

### Lotte-Science

```bash
data_dir="./data/datasets/lotte/science/test"
backbone_name_or_path="jingtao/Dense-bert_base-distil-msmarco"
output_dir="./data/dense-mlm/english-marco/inference_baseline/selfdr-distil/lotte/science/test"

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
    --pooling average \
    --similarity_metric ip \
    --topk 1000
```

The output is:
```python
{'NDCG@1': 0.17898, 'NDCG@3': 0.14893, 'NDCG@5': 0.14468, 'NDCG@10': 0.14355, 'NDCG@100': 0.18808}
{'MAP@1': 0.03643, 'MAP@3': 0.06071, 'MAP@5': 0.07301, 'MAP@10': 0.08439, 'MAP@100': 0.09808}
{'Recall@10': 0.13761, 'Recall@50': 0.23473, 'Recall@100': 0.28347, 'Recall@200': 0.34129, 'Recall@500': 0.41175, 'Recall@1000': 0.46811}
{'P@1': 0.17898, 'P@3': 0.13237, 'P@5': 0.11512, 'P@10': 0.08265, 'P@100': 0.01861}
{'MRR@10': 0.24748, 'MRR@100': 0.25591}
```

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