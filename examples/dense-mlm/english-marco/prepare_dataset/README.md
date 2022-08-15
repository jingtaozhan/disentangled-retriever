# English Retrieval Datasets

## MS MARCO Passage Ranking

MS MARCO passage ranking dataset contains passages from web pages and queries from search logs. It consists of 0.5 million training queries and 8.8 million passages. **We use it as training data.**

```bash
sh ./examples/dense-mlm/english-marco/prepare_dataset/prepare_msmarco.sh
```

```bash
cd data/datasets/msmarco-passage
wget https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz
wget https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz
cd -
python ./examples/dense-mlm/english-marco/prepare_dataset/prepare_marco_hardneg.py data/datasets/msmarco-passage/msmarco-hard-negatives.jsonl.gz data/datasets/msmarco-passage/msmarco-hard-negatives.tsv
```

## Lotte

Lotte collects questions and answers posted on StackExchange and divides them into five topics including writing, recreation, science, technology, and lifestyle. It regards the accepted or upvoted answers as relevant.  **We use the five sub-datasets for out-of-domain evaluation.**
```bash
pip install --upgrade ir_datasets
python ./examples/dense-mlm/english-marco/prepare_dataset/prepare_lotte.py ./data/datasets/lotte
```

## TREC-Covid

TREC-Covid is a retrieval dataset about searching COVID-19-related information from biomedical literature articles. The original annotation is biased towards lexical retrieval methods and we utilize a more comprehensive annotation released by Thakur et al. **We use it for out-of-domain evaluation.**
```bash
cd data/datasets
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid-beir.zip   
unzip trec-covid-beir.zip 
mv trec-covid-beir trec-covid

cd -
python ./examples/dense-mlm/english-marco/prepare_dataset/prepare_trec-covid.py data/datasets/trec-covid data/datasets/trec-covid
```