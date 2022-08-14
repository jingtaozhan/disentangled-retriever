# Chinese Retrieval Datasets

## Dureader

Dureader contains passages from web pages and queries from search logs. It is originally proposed as a Question-Answering dataset. We transform it into a retrieval dataset by treating the passage labeled `most related to the query' as the positive passage. **We use it as the training data.**

First, download Dureader dataset.
```bash
sh ./examples/dense-mlm/chinese-dureader/prepare_dataset/download_dureader.sh
```
Next, transform the original Question-Answering dataset into a retrieval dataset. The questions are queries, and the `most related' paragraphs are relevant passages. We prepend the titles to the begining of the passages.
```
python ./examples/dense-mlm/chinese-dureader/prepare_dataset/process_dureader.py ./data/datasets/dureader/preprocessed ./data/datasets/dureader
```
The dataset is saved in `data/datasets/dureader'.

## CPR Benchmark

CPR benchmark consists of three human-annotated domain-specific retrieval datasets collected from an e-commerce platform (Taobao), a video platform (Youku), and medical search within a search engine (Quark). **We use them to evaluate out-of-domain performance.**
```bash
cd data/datasets

# Download CPR datasets
git clone git@github.com:Alibaba-NLP/Multi-CPR.git
cp -r ./Multi-CPR/data/ecom ./cpr-ecom
cp -r ./Multi-CPR/data/medical ./cpr-medical
cat ./cpr-medical/corpus_split_*.tsv > ./cpr-medical/corpus.tsv
cp -r ./Multi-CPR/data/video ./cpr-video

# Now we create symbolic links to unify the filenames 
for domain in "cpr-ecom" "cpr-medical" "cpr-video"
do
    cd $domain
    ln -s train.query.txt query.train
    ln -s qrels.train.tsv qrels.train
    ln -s dev.query.txt query.dev
    ln -s qrels.dev.tsv qrels.dev
    cd ..
done 
```

## cMedQA v2

cMedQAv2 is constructed based on an online Chinese medical question answering forum. It collects user descriptions of their symptoms and the diagnosis or suggestions responded by doctors. We regard user descriptions as queries and responses as relevant documents. **We use it to evaluate out-of-domain performance.**

```bash
mkdir -p data/datasets/cmedqav2
cd data/datasets/cmedqav2
# download dataset
git clone git@github.com:zhangsheng93/cMedQA2.git
cd cMedQA2
unzip answer.zip
unzip question.zip
unzip train_candidates.zip
unzip dev_candidates.zip 
unzip test_candidates.zip 

cd ../../../..
# process 
python ./examples/dense-mlm/chinese-dureader/prepare_dataset/process_cmedqav2.py ./data/datasets/cmedqav2/cMedQA2 ./data/datasets/cmedqav2 
```