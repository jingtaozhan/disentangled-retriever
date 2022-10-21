# Disentangled Neural Ranking

<p align="left">
    <a href="https://github.com/jingtaozhan/disentangled-retriever">
    <img alt="Disentangle" src="https://img.shields.io/badge/NeuralIR-Disentangle-blueviolet">
    </a>
    <a href="https://github.com/jingtaozhan/disentangled-retriever/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg">
    </a>
    <a>
    <a href="https://pytorch.org">
    <img alt="made-with-pytorch" src="https://img.shields.io/badge/Made%20with-Pytorch-red.svg">
    </a>
    <a>
    <a href="https://github.com/jingtaozhan/disentangled-retriever">
    <img alt="code-size" src="https://img.shields.io/github/languages/code-size/jingtaozhan/disentangled-retriever?color=green">
    </a>
    <a href="https://github.com/jingtaozhan/disentangled-retriever">
    </a>
</p>

This is the official repo for our paper [Disentangled Modeling of Domain and Relevance for Adaptable Dense Retrieval](https://arxiv.org/pdf/2208.05753.pdf). Disentangled Neural Ranking is a novel paradigm that supports effective and flexible domain adaptation for neural ranking models including Dense Retrieval, uniCOIL, SPLADE, ColBERT, and BERT re-ranker.

## Features

- Various ranking architectures, including Dense Retrieval, uniCOIL, SPLADE, ColBERT, and BERT re-ranker. 
- Two source-domain finetuning methods, contrastive finetuning and distillation. 
- *One command line for unsupervised and effective domain adaption.*
- *One command line for effective few-shot domain adaption.*
- Huggingface-style training and inference, supporting multi-gpus, mixed precision, etc.


## Quick Links

- [Quick Tour](#quick-tour)
- [Installation](#installation)
- [Released Models](#released-models)
- [Example Usage](#example-usage)
- [Preparing Datasets](#preparing-datasets)
- [Unsupervisedly Adapting to an Unseen Domain](#unsupervisedly-adapting-to-an-unseen-domain)
- [Learning Generic Relevance Estimation Ability](#learning-generic-relevance-estimation-ability)
- [Reproducing Results with Released Checkpoints](#reproducing-results-with-released-checkpoints)
- [Training Vanilla Neural Ranking Models](#training-vanilla-neural-ranking-models)
    

## Quick Tour

Neural Ranking models are vulnerable to domain shift: the trained models may even perform worse than traditional retrieval methods like BM25 in out-of-domain scenarios.

In this work, we propose **Disentangled Neural Ranking (DNR)** to support effective and flexible domain adaptation. 
DNR consists of a Relevance Estimation Module (REM) for modeling domain-invariant matching patterns and several Domain Adaption Modules (DAMs) for modeling domain-specific features of multiple target corpora. 
DNR enables a flexible training paradigm in which REM is trained with supervision once and DAMs are trained with unsupervised data. 

Neural Ranking   |  Disentangled Neural Ranking
:-------------------------:|:-------------------------:
<img src="./figures/dr-modeling.png" height="80%">  | <img src="./figures/ddr-modeling.png" height="80%"> 

The idea of DNR can date back to classic retrieval models in the pre-neural-ranking era. BM25 utilizes the same formula for estimating relevance scores across domains but measures word importance with corpus-specific IDF values. 
However, it does not exist in vanilla neural ranking models where the abilities of relevance estimation and domain modeling are jointly learned during training and entangled within the model parameters. 

Here are two examples when we apply disentangled modeling for domain adaption. We plot the figure where y-axis shows the relative improvement over BM25 and x-axis shows different out-of-domain test sets.
The ranking performance of Dense Retrieval (DR) and Disentangled Dense Retrieval (DDR) is shown below.  
NDCG@10   |  Recall@1000
:-------------------------:|:-------------------------:
<img src="./figures/NDCG@10.png" width="100%"> | <img src="./figures/R@1000.png" width="97%"> 

The ranking performance of ColBERT and Disentangled ColBERT (D-ColBERT) is shown below. 
NDCG@10   |  Recall@1000
:-------------------------:|:-------------------------:
<img src="./figures/colbert-ndcg@10.png" width="100%"> | <img src="./figures/colbert-recall@1000.png" width="97%"> 

Disentangled modeling brings amazing out-of-domain performance gains!
More details are available in our [paper](https://arxiv.org/pdf/2208.05753.pdf).

## Installation

This repo is developed with [PyTorch]((https://pytorch.org/get-started/locally/)) and [Faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md). They should be installed manually due to the requirement of platform-specific custom configuration. In our development, we run the following commands for installation.
```bash
# XX.X is a placeholder for cudatoolkit version. It should be specified according to your environment
conda install pytorch torchvision torchaudio cudatoolkit=XX.X -c pytorch 
conda install -c conda-forge faiss-gpu
```

After these, now you can install from our code: 
```bash
git clone https://github.com/jingtaozhan/disentangled-retriever
cd disentangled-retriever
pip install .
```
For development, use
```
pip install --editable .
```

## Released Models

We release about 50 models to facilitate reproducibility and reusage. You do not have to manually download these. **They will be automatically downloaded at runtime.**

<details>
  <summary> Relevance Estimation Modules for Dense Retrieval (<i>click to expand</i>)</summary>

- (English) Contrastively trained on MS MARCO Passage Ranking: https://huggingface.co/jingtao/REM-bert_base-dense-contrast-msmarco/resolve/main/lora192-pa4.zip
- (English) Knowledge distilled on MS MARCO Passage Ranking: https://huggingface.co/jingtao/REM-bert_base-dense-distil-msmarco/resolve/main/lora192-pa4.zip
- (Chinese) Contrastively trained on Dureader: https://huggingface.co/jingtao/REM-bert_base-dense-contrast-dureader/resolve/main/lora192-pa4.zip
- (Chinese) Knowledge distilled on Dureader: https://huggingface.co/jingtao/REM-bert_base-dense-distil-dureader/resolve/main/lora192-pa4.zip

</details>

<details>
  <summary> Relevance Estimation Modules for UniCOIL (<i>click to expand</i>)</summary>

- (English) Contrastively trained on MS MARCO Passage Ranking: https://huggingface.co/jingtao/REM-bert_base-unicoil-contrast-msmarco/resolve/main/lora192-pa4.zip
- (English) Knowledge distilled on MS MARCO Passage Ranking: https://huggingface.co/jingtao/REM-bert_base-unicoil-distil-msmarco/resolve/main/lora192-pa4.zip
- (Chinese) Contrastively trained on Dureader: https://huggingface.co/jingtao/REM-bert_base-unicoil-contrast-dureader/resolve/main/lora192-pa4.zip
- (Chinese) Knowledge distilled on Dureader: https://huggingface.co/jingtao/REM-bert_base-unicoil-distil-dureader/resolve/main/lora192-pa4.zip

</details>

<details>
  <summary> Relevance Estimation Modules for SPLADE (<i>click to expand</i>)</summary>

- (English) Contrastively trained on MS MARCO Passage Ranking: https://huggingface.co/jingtao/REM-bert_base-splade-contrast-msmarco/resolve/main/lora192-pa4.zip
- (English) Knowledge distilled on MS MARCO Passage Ranking: https://huggingface.co/jingtao/REM-bert_base-splade-distil-msmarco/resolve/main/lora192-pa4.zip
- (Chinese) Contrastively trained on Dureader: https://huggingface.co/jingtao/REM-bert_base-splade-contrast-dureader/resolve/main/lora192-pa4.zip
- (Chinese) Knowledge distilled on Dureader: https://huggingface.co/jingtao/REM-bert_base-splade-distil-dureader/resolve/main/lora192-pa4.zip

</details>

<details>
  <summary> Relevance Estimation Modules for ColBERT (<i>click to expand</i>)</summary>

- (English) Contrastively trained on MS MARCO Passage Ranking: https://huggingface.co/jingtao/REM-bert_base-colbert-contrast-msmarco/resolve/main/lora192-pa4.zip
- (English) Knowledge distilled on MS MARCO Passage Ranking: https://huggingface.co/jingtao/REM-bert_base-colbert-distil-msmarco/resolve/main/lora192-pa4.zip
- (Chinese) Contrastively trained on Dureader: https://huggingface.co/jingtao/REM-bert_base-colbert-contrast-dureader/resolve/main/lora192-pa4.zip
- (Chinese) Knowledge distilled on Dureader: https://huggingface.co/jingtao/REM-bert_base-colbert-distil-dureader/resolve/main/lora192-pa4.zip

</details>

<details>
  <summary> Relevance Estimation Modules for BERT re-ranker (<i>click to expand</i>)</summary>

- (English) Trained on MS MARCO Passage Ranking: https://huggingface.co/jingtao/REM-bert_base-rerank-msmarco/resolve/main/lora192-pa4.zip
- (Chinese) Trained on Dureader: https://huggingface.co/jingtao/REM-bert_base-rerank-dureader/resolve/main/lora192-pa4.zip

</details>

<details>
  <summary> Domain Adaption Modules for various datasets  (<i>click to expand</i>)</summary>

- Trained on English test sets (Masked Language Modeling)
    - Trained on MS MARCO Passage corpus: [jingtao/DAM-bert_base-mlm-msmarco](https://huggingface.co/jingtao/DAM-bert_base-mlm-msmarco)
    - Trained on TREC-Covid corpus: [jingtao/DAM-bert_base-mlm-msmarco-trec_covid](https://huggingface.co/jingtao/DAM-bert_base-mlm-msmarco-trec_covid)
    - Trained on Lotte-Writing corpus: [jingtao/DAM-bert_base-mlm-msmarco-lotte_write_test](https://huggingface.co/jingtao/DAM-bert_base-mlm-msmarco-lotte_write_test)
    - Trained on Lotte-Recreation corpus: [jingtao/DAM-bert_base-mlm-msmarco-lotte_rec_test](https://huggingface.co/jingtao/DAM-bert_base-mlm-msmarco-lotte_rec_test)
    - Trained on Lotte-Technology corpus: [jingtao/DAM-bert_base-mlm-msmarco-lotte_tech_test](https://huggingface.co/jingtao/DAM-bert_base-mlm-msmarco-lotte_tech_test)
    - Trained on Lotte-Lifestyle corpus: [jingtao/DAM-bert_base-mlm-msmarco-lotte_life_test](https://huggingface.co/jingtao/DAM-bert_base-mlm-msmarco-lotte_life_test)
    - Trained on Lotte-Science corpus: [jingtao/DAM-bert_base-mlm-msmarco-lotte_sci_test](https://huggingface.co/jingtao/DAM-bert_base-mlm-msmarco-lotte_sci_test)
- Trained on Chinese test sets (Masked Language Modeling)
    - Trained on Dureader corpus: [jingtao/DAM-bert_base-mlm-dureader](https://huggingface.co/jingtao/DAM-bert_base-mlm-dureader)
    - Trained on CPR-Ecommerce corpus: [jingtao/DAM-bert_base-mlm-dureader-cpr_ecom](https://huggingface.co/jingtao/DAM-bert_base-mlm-dureader-cpr_ecom)
    - Trained on CPR-Video corpus: [jingtao/DAM-bert_base-mlm-dureader-cpr_video](https://huggingface.co/jingtao/DAM-bert_base-mlm-dureader-cpr_video)
    - Trained on CPR-Medical corpus: [jingtao/DAM-bert_base-mlm-dureader-cpr_medical](https://huggingface.co/jingtao/DAM-bert_base-mlm-dureader-cpr_medical)
    - Trained on cMedQAv2 corpus: [jingtao/DAM-bert_base-mlm-dureader-cmedqav2](https://huggingface.co/jingtao/DAM-bert_base-mlm-dureader-cmedqav2)
    
</details>


Besides Disentangled Neural Ranking models, we also release the vanilla/traditional neural ranking models, which are baselines in our paper.
<details>
  <summary> Vanilla Neural Ranking Checkpoints (<i>click to expand</i>)</summary>

<details>
  <summary> Vanilla Dense Retrieval (<i>click to expand</i>)</summary>

- (English) Contrastively trained on MS MARCO Passage Ranking: [jingtao/Dense-bert_base-contrast-msmarco](https://huggingface.co/jingtao/Dense-bert_base-contrast-msmarco)
- (English) Knowledge distilled on MS MARCO Passage Ranking: [jingtao/Dense-bert_base-distil-msmarco](https://huggingface.co/jingtao/Dense-bert_base-distil-msmarco)
- (Chinese) Contrastively trained on Dureader: [jingtao/Dense-bert_base-contrast-dureader](https://huggingface.co/jingtao/Dense-bert_base-contrast-dureader)
- (Chinese) Knowledge distilled on Dureader: [jingtao/Dense-bert_base-distil-dureader](https://huggingface.co/jingtao/Dense-bert_base-distil-dureader)


</details>

<details>
  <summary> Vanilla uniCOIL (<i>click to expand</i>)</summary>

- (English) Contrastively trained on MS MARCO Passage Ranking: [jingtao/uniCOIL-bert_base-contrast-msmarco](https://huggingface.co/jingtao/uniCOIL-bert_base-contrast-msmarco)
- (English) Knowledge distilled on MS MARCO Passage Ranking: [jingtao/uniCOIL-bert_base-distil-msmarco](https://huggingface.co/jingtao/uniCOIL-bert_base-distil-msmarco)
- (Chinese) Contrastively trained on Dureader: [jingtao/uniCOIL-bert_base-contrast-dureader](https://huggingface.co/jingtao/uniCOIL-bert_base-contrast-dureader)
- (Chinese) Knowledge distilled on Dureader: [jingtao/uniCOIL-bert_base-distil-dureader](https://huggingface.co/jingtao/uniCOIL-bert_base-distil-dureader)

</details>

<details>
  <summary> Vanilla SPLADE (<i>click to expand</i>)</summary>

- (English) Contrastively trained on MS MARCO Passage Ranking: [jingtao/SPLADE-bert_base-contrast-msmarco](https://huggingface.co/jingtao/SPLADE-bert_base-contrast-msmarco)
- (English) Knowledge distilled on MS MARCO Passage Ranking: [jingtao/SPLADE-bert_base-distil-msmarco](https://huggingface.co/jingtao/SPLADE-bert_base-distil-msmarco)
- (Chinese) Contrastively trained on Dureader: [jingtao/SPLADE-bert_base-contrast-dureader](https://huggingface.co/jingtao/SPLADE-bert_base-contrast-dureader)
- (Chinese) Knowledge distilled on Dureader: [jingtao/SPLADE-bert_base-distil-dureader](https://huggingface.co/jingtao/SPLADE-bert_base-distil-dureader)

</details>

<details>
  <summary> Vanilla ColBERT (<i>click to expand</i>)</summary>

- (English) Contrastively trained on MS MARCO Passage Ranking: [jingtao/ColBERT-bert_base-contrast-msmarco](https://huggingface.co/jingtao/ColBERT-bert_base-contrast-msmarco)
- (English) Knowledge distilled on MS MARCO Passage Ranking: [jingtao/ColBERT-bert_base-distil-msmarco](https://huggingface.co/jingtao/ColBERT-bert_base-distil-msmarco)
- (Chinese) Contrastively trained on Dureader: [jingtao/ColBERT-bert_base-contrast-dureader](https://huggingface.co/jingtao/ColBERT-bert_base-contrast-dureader)
- (Chinese) Knowledge distilled on Dureader: [jingtao/ColBERT-bert_base-distil-dureader](https://huggingface.co/jingtao/ColBERT-bert_base-distil-dureader)

</details>

<details>
  <summary> Vanilla BERT re-ranker (<i>click to expand</i>)</summary>

- (English) Trained on MS MARCO Passage Ranking: [jingtao/Rerank-bert_base-msmarco](https://huggingface.co/jingtao/Rerank-bert_base-msmarco)
- (Chinese) Trained on Dureader:[jingtao/Rerank-bert_base-dureader](https://huggingface.co/jingtao/Rerank-bert_base-dureader)

</details>


</details>
*Note: Our code also supports training and evaluating vanilla neural ranking models!*

## Example usage:

Here is an example about using disentangled dense retrieval for ranking. The REM is generic, while the DAM is domain-specifically trained to mitigate the domain shift. The two modules are assembled during inference.
```python
from transformers import AutoConfig, AutoTokenizer
from disentangled_retriever.dense.modeling import AutoDenseModel

# This is the Relevance Estimation Module (REM) contrastively trained on MS MARCO
# It can be used in various English domains.
REM_URL = "https://huggingface.co/jingtao/REM-bert_base-dense-contrast-msmarco/resolve/main/lora192-pa4.zip"
## For example, we will apply the model to TREC-Covid dataset. 
# Here is the Domain Adaption Module for this dataset.
DAM_NAME = "jingtao/DAM-bert_base-mlm-msmarco-trec_covid"

## Load the modules
config = AutoConfig.from_pretrained(DAM_NAME)
config.similarity_metric, config.pooling = "ip", "average"
tokenizer = AutoTokenizer.from_pretrained(DAM_NAME, config=config)
model = AutoDenseModel.from_pretrained(DAM_NAME, config=config)
adapter_name = model.load_adapter(REM_URL)
model.set_active_adapters(adapter_name)
model.merge_lora(adapter_name)

## Let's try to compute the similarities
queries  = ["When will the COVID-19 pandemic end?", "What are the impacts of COVID-19 pandemic to society?"]
passages = ["It will end soon.", "It makes us care for each other."]
query_embeds = model(**tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512))
passage_embeds = model(**tokenizer(passages, return_tensors="pt", padding=True, truncation=True, max_length=512))

print(query_embeds @ passage_embeds.T)
```
Results are:
```python
tensor([[107.6821, 101.4270],
        [103.7373, 105.0448]], grad_fn=<MmBackward0>)
```

## Preparing datasets

We will use various datasets to show how disentangled modeling facilitates flexible domain adaption. Before the demonstration, please download and preprocess the corresponding datasets. Here we provide detailed instructions:

- [Preparing English training data and out-of-domain test sets](./examples/dense-mlm/english-marco/prepare_dataset/README.md)
- [Preparing Chinese training data and out-of-domain test sets](./examples/dense-mlm/chinese-dureader/prepare_dataset/README.md)


## Unsupervisedly Adapting to an Unseen Domain

Suppose you already have a REM module (trained by yourself or provided by us) and you need to adapt the model to an unseen domain. To do this, just train a Domain Adaption Module (DAM) to mitigate the domain shift. 

The training process is completely unsupervised and only requires the target-domain corpus. Each line of the corpus file should be formatted as `id doc' separated by tab. Then you can train a DAM model with only one command. 
```bash
python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.adapt.run_adapt_with_mlm \
    --corpus_path ... ... ...
```
The trained DAM can be combined with different REMs and formalize a well-performing neural ranking models. We provide many REMs (see this [section](#released-models)) that correspond to different ranking methods or are trained with different losses. The trained DAM can be combined with any REM to become an effective ranking model, e.g., a Dense Retrievla model or a ColBERT model. 
For example, if you want to acquire a dense retrieval model, use the following command for inference:
```bash
python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.dense.evaluate.run_eval \
    --backbone_name_or_path [path-to-the-trained-DAM] \
    --adapter_name_or_path [path-to-the-dense-retrieval-rem] \
    --corpus_path ... --query_path ... ... ...
```
If you want to acquire a ColBERT model, use the following command for inference:
```bash
python -m torch.distributed.launch --nproc_per_node 4 \
    -m disentangled_retriever.colbert.evaluate.run_eval \
    --backbone_name_or_path [path-to-the-trained-DAM] \
    --adapter_name_or_path [path-to-the-dense-retrieval-rem] \
    --corpus_path ... --query_path ... ... ...
```

We give an adaption example [here](./examples/dense-mlm/english-marco/adapt_domain.md) with detailed instructions. Please trying this example before using our methods on your own datasets.

## Learning Generic Relevance Estimation Ability

We already release a bunch of Relevance Estimation Modules (REMs) for various kinds of ranking methods. You can directly adopt these public checkpoints. 
But if you have some private labeled data and want to a Relevance Estimation Module (REM) on it, here we provide instructions on how to do this.

To directly use this codebase for training, you need to convert your dataformat as follows
- corpus.tsv: corpus file. each line is `docid doc' separated by tab.
- query.train: training queries. each line is `qid query' separated by tab.
- qrels.train: annotations. each line is `qid 0 docid rel_level' separated by tab. 
- [Optional] hard negative file for contrastive training: each line is `qid   neg_docid1 neg_docid2 ...'. qid and neg_docids are separated by tab. neg_docids are separated by space.
- [Optional] soft labels for knowledge distillation: a pickle file containing a dict: {qid: {docid: score}}. It should contain the soft labels of positive pairs and of several negative pairs.

If you still have questions about the data formatting, you can check [how we convert MS MARCO](./examples/dense-mlm/english-marco/prepare_dataset/README.md#ms-marco-passage-ranking).

With formatted supervised data, now you can train a REM module. We use a disentangled finetuning trick: first training a DAM module to capture domain-specific features and then training the REM module to learn domain-invariant matching patterns. 
We provide two training examples, [one](./examples/dense-mlm/english-marco/train_rem.md) on English MS MARCO and [one](./examples/dense-mlm/chinese-dureader/train_rem.md) on Chinese Dureader. 

## Reproducing Results with Released Checkpoints

We provide commands for reproducing the various results in our [paper](https://arxiv.org/pdf/2208.05753.pdf).
- Evaluating Disentangled Dense Retrieval
  - [Evaluating Disentangled Dense Retrieval on English out-of-domain datasets](./examples/dense-mlm/english-marco/inference.md)
  - [Evaluating Disentangled Dense Retrieval on Chinese out-of-domain datasets](./examples/dense-mlm/chinese-dureader/inference.md)
- Evaluating Disentangled uniCOIL
- Evaluating Disentangled SPLADE
- Evaluating Disentangled ColBERT
- Evaluating Disentangled BERT re-ranker


<!-- ### Generic Relevance Estimation: Training REM 

Relevance Estimation Module (REM) is only required to trained once. In our paper, we use MS MARCO as English training data and use Dureader as Chinese training data. We also explore two finetuning methods. The commands are provided. You can also use your own supervised data to train a REM module. 

- [Training REM on MS MARCO](./examples/dense-mlm/english-marco/train_rem.md)
    - Disentangled Finetuning
    - Hard negatives, contrastive loss. or Cross-encoder scores, Margin-MSE loss.
- [Training REM on Dureader](./examples/dense-mlm/chinese-dureader/train_rem.md)
    - Disentangled Finetuning
    - Inbatch negatives, contrastive loss.

### Unsupervised Domain Adaption: Training DAM

With a ready REM module, Domain Adaption Module (DAM) is trained unsupervisedly for each target domain to mitigate domain shift. In the following, we give examples about how to adapt to an unseen English/Chinese domain. Note that we suppose the REM module is ready and directly load the trained REM module. 

- [Adapting to an unseen English domain](./examples/dense-mlm/english-marco/adapt_domain.md)
    - Unsupervised Adaption yet high effectiveness.
    - Take Lotte-Technology as an example.
- [Adapting to an unseen Chinese domain](./examples/dense-mlm/chinese-dureader/adapt_domain.md)
    - Unsupervised Adaption yet high effectiveness.
    - Take CPR-Ecommerce as an example. -->

## Training Vanilla Neural Ranking Models

This powerful codebase not only supports Disentangled Neural Ranking, but also vanilla Neural Ranking models. You can easily reproduce state-of-the-art Dense Retrieval, uniCOIL, SPLADE, ColBERT, and BERT rerankers using this codebase! The instructions are provided as below.

- Dense Retrieval
  - [Training Dense Retrieval models on MS MARCO](./examples/dense-mlm/english-marco/train_baseline.md)
  - [Training Dense Retrieval models on Dureader](./examples/dense-mlm/chinese-dureader/train_baseline.md)
  - [Evaluating Dense Retrieval models on English out-of-domain datasets](./examples/dense-mlm/english-marco/inference_baseline.md)
  - [Evaluating Dense Retrieval models on Chinese out-of-domain datasets](./examples/dense-mlm/chinese-dureader/inference_baseline.md)
- uniCOIL
- SPLADE
- ColBERT
- BERT Re-rankers


## Citation
If you find our work useful, please consider citing us :)
```
@article{zhan2022disentangled,
  title={Disentangled Modeling of Domain and Relevance for Adaptable Dense Retrieval},
  author={Zhan, Jingtao and Ai, Qingyao and Liu, Yiqun and Mao, Jiaxin and Xie, Xiaohui and Zhang, Min and Ma, Shaoping},
  journal={arXiv preprint arXiv:2208.05753},
  year={2022}
}
```