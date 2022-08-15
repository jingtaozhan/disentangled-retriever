# Inference with Trained Models

Note: Different environments, e.g., different GPUs or different systems, may result in **slightly** different metric numbers. 

## TREC-Covid

<details>
<summary>Inference with unsupervised DAM and contrastively trained REM.</summary>

```bash
data_dir="./data/datasets/trec-covid"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-msmarco-trec_covid"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-contrast-msmarco/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/english-marco/inference/trec-covid/contrast"

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
    --similarity_metric ip \
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.79, 'NDCG@3': 0.75335, 'NDCG@5': 0.75797, 'NDCG@10': 0.72016, 'NDCG@100': 0.50657}
{'MAP@1': 0.00214, 'MAP@3': 0.00554, 'MAP@5': 0.00899, 'MAP@10': 0.01589, 'MAP@100': 0.08222}
{'Recall@10': 0.01826, 'Recall@50': 0.06965, 'Recall@100': 0.11583, 'Recall@200': 0.18723, 'Recall@500': 0.31038, 'Recall@1000': 0.43154}
{'P@1': 0.86, 'P@3': 0.8, 'P@5': 0.8, 'P@10': 0.758, 'P@100': 0.5126}
{'MRR@10': 0.9065, 'MRR@100': 0.9065}
```
</details>


<details>
<summary>Inference with unsupervised DAM and distilled REM.</summary>

```bash
data_dir="./data/datasets/trec-covid"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-msmarco-trec_covid"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-distil-msmarco/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/english-marco/inference/trec-covid/distil"

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
    --similarity_metric ip \
    --topk 1000 \
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.74, 'NDCG@3': 0.76184, 'NDCG@5': 0.74559, 'NDCG@10': 0.71414, 'NDCG@100': 0.5255}
{'MAP@1': 0.00213, 'MAP@3': 0.00584, 'MAP@5': 0.00916, 'MAP@10': 0.01634, 'MAP@100': 0.08932}
{'Recall@10': 0.01837, 'Recall@50': 0.07276, 'Recall@100': 0.12162, 'Recall@200': 0.19344, 'Recall@500': 0.31846, 'Recall@1000': 0.43381}
{'P@1': 0.84, 'P@3': 0.82667, 'P@5': 0.812, 'P@10': 0.76, 'P@100': 0.5392}
{'MRR@10': 0.90833, 'MRR@100': 0.90833}
```
</details>


## Lotte-Writing

<details>
<summary>Inference with unsupervised DAM and contrastively trained REM.</summary>

```bash
data_dir="./data/datasets/lotte/writing/test"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-msmarco-lotte_write_test"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-contrast-msmarco/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/english-marco/inference/lotte/writing/test/contrast"

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
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.5095, 'NDCG@3': 0.45169, 'NDCG@5': 0.43616, 'NDCG@10': 0.44691, 'NDCG@100': 0.51494}
{'MAP@1': 0.12147, 'MAP@3': 0.2373, 'MAP@5': 0.28831, 'MAP@10': 0.33316, 'MAP@100': 0.3637}
{'Recall@10': 0.45349, 'Recall@50': 0.60805, 'Recall@100': 0.66289, 'Recall@200': 0.71072, 'Recall@500': 0.76843, 'Recall@1000': 0.80531}
{'P@1': 0.5095, 'P@3': 0.40633, 'P@5': 0.3334, 'P@10': 0.22665, 'P@100': 0.03588}
{'MRR@10': 0.61338, 'MRR@100': 0.61897}
```
</details>

<details>
<summary>Inference with unsupervised DAM and distilled REM.</summary>

```bash
data_dir="./data/datasets/lotte/writing/test"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-msmarco-lotte_write_test"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-distil-msmarco/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/english-marco/inference/lotte/writing/test/distil"

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
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.495, 'NDCG@3': 0.43645, 'NDCG@5': 0.42318, 'NDCG@10': 0.4352, 'NDCG@100': 0.50468}
{'MAP@1': 0.11768, 'MAP@3': 0.22946, 'MAP@5': 0.27811, 'MAP@10': 0.32088, 'MAP@100': 0.35231}
{'Recall@10': 0.44551, 'Recall@50': 0.60915, 'Recall@100': 0.65785, 'Recall@200': 0.70034, 'Recall@500': 0.75236, 'Recall@1000': 0.7906}
{'P@1': 0.495, 'P@3': 0.39317, 'P@5': 0.3239, 'P@10': 0.22035, 'P@100': 0.03524}
{'MRR@10': 0.59987, 'MRR@100': 0.60515}
```
</details>

## Lotte-Recreation

<details>
<summary>Inference with unsupervised DAM and contrastively trained REM.</summary>

```bash
data_dir="./data/datasets/lotte/recreation/test"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-msmarco-lotte_rec_test"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-contrast-msmarco/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/english-marco/inference/lotte/recreation/test/contrast"

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
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.41409, 'NDCG@3': 0.37917, 'NDCG@5': 0.3857, 'NDCG@10': 0.41137, 'NDCG@100': 0.48288}
{'MAP@1': 0.16883, 'MAP@3': 0.26784, 'MAP@5': 0.29883, 'MAP@10': 0.32334, 'MAP@100': 0.34635}
{'Recall@10': 0.46254, 'Recall@50': 0.64233, 'Recall@100': 0.70079, 'Recall@200': 0.76026, 'Recall@500': 0.82539, 'Recall@1000': 0.86558}
{'P@1': 0.41409, 'P@3': 0.28971, 'P@5': 0.22328, 'P@10': 0.14276, 'P@100': 0.02317}
{'MRR@10': 0.51379, 'MRR@100': 0.52079}
```
</details>

<details>
<summary>Inference with unsupervised DAM and distilled REM.</summary>

```bash
data_dir="./data/datasets/lotte/recreation/test"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-msmarco-lotte_rec_test"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-distil-msmarco/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/english-marco/inference/lotte/recreation/test/distil"

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
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.44705, 'NDCG@3': 0.39072, 'NDCG@5': 0.39461, 'NDCG@10': 0.42432, 'NDCG@100': 0.49244}
{'MAP@1': 0.18487, 'MAP@3': 0.27942, 'MAP@5': 0.30814, 'MAP@10': 0.33525, 'MAP@100': 0.35706}
{'Recall@10': 0.47307, 'Recall@50': 0.63539, 'Recall@100': 0.70083, 'Recall@200': 0.75057, 'Recall@500': 0.81085, 'Recall@1000': 0.84896}
{'P@1': 0.44705, 'P@3': 0.29237, 'P@5': 0.22288, 'P@10': 0.14525, 'P@100': 0.02316}
{'MRR@10': 0.53626, 'MRR@100': 0.54228}
```
</details>

## Lotte-Technology

<details>
<summary>Inference with unsupervised DAM and contrastively trained REM.</summary>

```bash
data_dir="./data/datasets/lotte/technology/test"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-msmarco-lotte_tech_test"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-contrast-msmarco/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/english-marco/inference/lotte/technology/test/contrast"

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
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.26098, 'NDCG@3': 0.22058, 'NDCG@5': 0.2069, 'NDCG@10': 0.20652, 'NDCG@100': 0.28665}
{'MAP@1': 0.04649, 'MAP@3': 0.0827, 'MAP@5': 0.09894, 'MAP@10': 0.11555, 'MAP@100': 0.14113}
{'Recall@10': 0.19909, 'Recall@50': 0.36771, 'Recall@100': 0.44962, 'Recall@200': 0.52863, 'Recall@500': 0.6382, 'Recall@1000': 0.71094}
{'P@1': 0.26098, 'P@3': 0.20126, 'P@5': 0.16756, 'P@10': 0.12201, 'P@100': 0.02971}
{'MRR@10': 0.36382, 'MRR@100': 0.37449}
```
</details>

<details>
<summary>Inference with unsupervised DAM and distilled REM.</summary>

```bash
data_dir="./data/datasets/lotte/technology/test"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-msmarco-lotte_tech_test"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-distil-msmarco/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/english-marco/inference/lotte/technology/test/distil"

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
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.24501, 'NDCG@3': 0.21562, 'NDCG@5': 0.20208, 'NDCG@10': 0.20365, 'NDCG@100': 0.27992}
{'MAP@1': 0.04359, 'MAP@3': 0.08011, 'MAP@5': 0.0954, 'MAP@10': 0.11307, 'MAP@100': 0.13746}
{'Recall@10': 0.20083, 'Recall@50': 0.36018, 'Recall@100': 0.43769, 'Recall@200': 0.51528, 'Recall@500': 0.62207, 'Recall@1000': 0.69808}
{'P@1': 0.24501, 'P@3': 0.1996, 'P@5': 0.16567, 'P@10': 0.12136, 'P@100': 0.02897}
{'MRR@10': 0.35339, 'MRR@100': 0.36415}
```
</details>

## Lotte-Lifestyle

<details>
<summary>Inference with unsupervised DAM and contrastively trained REM.</summary>

```bash
data_dir="./data/datasets/lotte/lifestyle/test"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-msmarco-lotte_life_test"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-contrast-msmarco/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/english-marco/inference/lotte/lifestyle/test/contrast"

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
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.47303, 'NDCG@3': 0.42567, 'NDCG@5': 0.41467, 'NDCG@10': 0.43067, 'NDCG@100': 0.51979}
{'MAP@1': 0.13568, 'MAP@3': 0.24031, 'MAP@5': 0.28115, 'MAP@10': 0.31907, 'MAP@100': 0.35464}
{'Recall@10': 0.45447, 'Recall@50': 0.64626, 'Recall@100': 0.71953, 'Recall@200': 0.78385, 'Recall@500': 0.85868, 'Recall@1000': 0.9034}
{'P@1': 0.47303, 'P@3': 0.3668, 'P@5': 0.2974, 'P@10': 0.20285, 'P@100': 0.03557}
{'MRR@10': 0.58244, 'MRR@100': 0.58846}
```
</details>

<details>
<summary>Inference with unsupervised DAM and distilled REM.</summary>

```bash
data_dir="./data/datasets/lotte/lifestyle/test"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-msmarco-lotte_life_test"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-distil-msmarco/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/english-marco/inference/lotte/lifestyle/test/distil"

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
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.49051, 'NDCG@3': 0.43245, 'NDCG@5': 0.41884, 'NDCG@10': 0.43334, 'NDCG@100': 0.52251}
{'MAP@1': 0.14116, 'MAP@3': 0.24556, 'MAP@5': 0.2862, 'MAP@10': 0.32317, 'MAP@100': 0.35818}
{'Recall@10': 0.44968, 'Recall@50': 0.64553, 'Recall@100': 0.7167, 'Recall@200': 0.78204, 'Recall@500': 0.85349, 'Recall@1000': 0.89946}
{'P@1': 0.49051, 'P@3': 0.36913, 'P@5': 0.2968, 'P@10': 0.20165, 'P@100': 0.03526}
{'MRR@10': 0.59496, 'MRR@100': 0.6011}
```
</details>

## Lotte-Science

<details>
<summary>Inference with unsupervised DAM and contrastively trained REM.</summary>

```bash
data_dir="./data/datasets/lotte/science/test"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-msmarco-lotte_sci_test"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-contrast-msmarco/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/english-marco/inference/lotte/science/test/contrast"

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
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.1884, 'NDCG@3': 0.15798, 'NDCG@5': 0.15141, 'NDCG@10': 0.14932, 'NDCG@100': 0.20055}
{'MAP@1': 0.03742, 'MAP@3': 0.06282, 'MAP@5': 0.07511, 'MAP@10': 0.08629, 'MAP@100': 0.10172}
{'Recall@10': 0.14292, 'Recall@50': 0.25126, 'Recall@100': 0.30774, 'Recall@200': 0.37133, 'Recall@500': 0.45555, 'Recall@1000': 0.51997}
{'P@1': 0.1884, 'P@3': 0.13965, 'P@5': 0.11968, 'P@10': 0.08612, 'P@100': 0.02022}
{'MRR@10': 0.26173, 'MRR@100': 0.27109}
```
</details>

<details>
<summary>Inference with unsupervised DAM and distilled REM.</summary>

```bash
data_dir="./data/datasets/lotte/science/test"
backbone_name_or_path="jingtao/DAM-bert_base-mlm-msmarco-lotte_sci_test"
adapter_name_or_path="https://huggingface.co/jingtao/REM-bert_base-dense-distil-msmarco/resolve/main/lora192-pa4.zip"
output_dir="./data/dense-mlm/english-marco/inference/lotte/science/test/distil"

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
    --save_corpus_embed \
    --save_query_embed
```

The output is:
```python
{'NDCG@1': 0.18493, 'NDCG@3': 0.15794, 'NDCG@5': 0.15017, 'NDCG@10': 0.15052, 'NDCG@100': 0.20323}
{'MAP@1': 0.03846, 'MAP@3': 0.06341, 'MAP@5': 0.07537, 'MAP@10': 0.08776, 'MAP@100': 0.10347}
{'Recall@10': 0.14537, 'Recall@50': 0.25825, 'Recall@100': 0.31601, 'Recall@200': 0.37353, 'Recall@500': 0.46337, 'Recall@1000': 0.53085}
{'P@1': 0.18493, 'P@3': 0.1408, 'P@5': 0.1182, 'P@10': 0.08642, 'P@100': 0.02041}
{'MRR@10': 0.25915, 'MRR@100': 0.26946}
```
</details>