# Disentangled Neural Retrieval

This is the official repo for our paper [Disentangled Modeling of Domain and Relevance for Adaptable Dense Retrieval](https://arxiv.org/pdf/2208.05753.pdf). 

## Quick Tour

Common Dense Retrieval (DR) is vulnerable to domain shift: the trained DR models perform worse than traditional retrieval methods like BM25 in out-of-domain scenarios.

In this work, we propose a novel Dense Retrieval framework named Disentangled Dense Retrieval (DDR) to **support effective and flexible domain adaptation** for DR models. 
DDR consists of a Relevance Estimation Module (REM) for modeling domain-invariant matching patterns and **several Domain Adaption Modules (DAMs) for modeling domain-specific features of multiple target corpora to mitigate domain shift**. 
By making the REM and DAMs disentangled, DDR enables a flexible training paradigm in which REM is trained with supervision once and DAMs are trained with unsupervised data. 

Dense Retrieval   |  Disentangled Dense Retrieval
:-------------------------:|:-------------------------:
<img src="./figures/dr-modeling.png" height="80%">  | <img src="./figures/ddr-modeling.png" height="80%"> 

The idea of DDR can date back to classic retrieval models in the pre-dense-retrieval era. **Take BM25 as an example. It utilizes the same formula for estimating relevance scores across domains but measures word importance with corpus-specific IDF values**. 
However, it does not exist in DR where the abilities of relevance estimation and domain modeling are jointly learned during training and entangled within the model parameters. 


Please check our [paper](https://arxiv.org/pdf/2208.05753.pdf) to see the amazing out-of-domain performance gains brought by the disentangled modeling!

## Installation

Three special dependencies should be installed manually: disentangled-retriever depends on [PyTorch](https://pytorch.org/get-started/locally/) and [Faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md), which require platform-specific custom configuration. They are not listed in the requirements and the installation is left to you. In our development, we run the following commands for installation.
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch # 1.12.1 version
conda install -c conda-forge faiss-gpu # 1.7.2 version
```
disentangled-retriever also depends on [adapter-transformers](https://github.com/adapter-hub/adapter-transformers). Since the library is still in development and api is unstable, run the following code to install a certain version (commit).
```bash
git clone git@github.com:adapter-hub/adapter-transformers.git
cd adapter-transformers
git checkout 74dd021
pip install .
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

## Reproducing Results with Trained Checkpoints

We have provided commands for reproducing the various results in our [paper](https://arxiv.org/pdf/2208.05753.pdf).
- [Reproducing results of Disentangled Dense Retrieval on English out-of-domain datasets](./examples/dense-mlm/english-marco/inference.md)
    - On multiple datasets: TREC-Covid, Lotte-Writing, Lotte-Recreation, Lotte-Technology, Lotte-Lifestyle, and Lotte-Science.
    - Evaluting both contrastively trained and distilled models.
- [Reproducing results of Disentangled Dense Retrieval on Chinese out-of-domain datasets](./examples/dense-mlm/chinese-dureader/inference.md)
    - On multiple datasets: CPR-Ecom, CPR-Video, CPR-Medical, cMedQAv2.
    - Evaluting both contrastively trained and distilled models.
- [Reproducing results of Dense Retrieval baselines on English out-of-domain datasets](./examples/dense-mlm/english-marco/inference_baseline.md)
    - Evaluating self-implemented Dense Retrieval baselines that follow the same finetuning settings as Disentangled Dense Retrieval. 
    - Evaluating (a) strong Dense Retrieval model(s) in the literature.
- [Reproducing results of Dense Retrieval baselines on Chinese out-of-domain datasets](./examples/dense-mlm/chinese-dureader/inference_baseline.md)
    - Evaluating self-implemented Dense Retrieval baselines that follow the same finetuning settings as Disentangled Dense Retrieval. 


## Unsupervised Domain Adaption: Training DAM

We will release the code soon! Please stay tuned.

## Generic Relevance Estimation: Training REM 

We will release the code soon! Please stay tuned.

## Training Dense Retrieval baselines

We will release the code soon! Please stay tuned.

