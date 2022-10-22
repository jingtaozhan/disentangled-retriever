import os
import sys
from tqdm import tqdm
from collections import defaultdict

def format_corpus():
    f = open(os.path.join(input_dir, "answer.csv"))
    f.readline()
    with open(os.path.join(output_dir, 'corpus.tsv'), 'w') as output_file:
        for line in f:
            ans_id, qid, content = line.split(",", maxsplit=2)
            output_file.write(f"{ans_id}\t{content.strip()}\n")


def format_train():
    qrels = defaultdict(set)
    f = open(os.path.join(input_dir, f"train_candidates.txt"))
    f.readline()
    for line in f:
        qid, ansid, _ = line.split(",")
        qrels[qid].add(ansid)
    
    f = open(os.path.join(input_dir, f"question.csv"))
    f.readline()
    with open(os.path.join(output_dir, "query.train"), 'w') as query_output, \
            open(os.path.join(output_dir, "qrels.train"), 'w') as qrel_output:
        for line in f:
            qid, query = line.split(",", maxsplit=1)
            query = query.strip()
            if qid in qrels:
                query_output.write(f"{qid}\t{query}\n")
                for ansid in qrels[qid]:
                    qrel_output.write(f"{qid}\t0\t{ansid}\t1\n")
                

def format_test_queries(mode):
    assert mode in ["dev", "test"]
    
    qrels = defaultdict(set)
    f = open(os.path.join(input_dir, f"{mode}_candidates.txt"))
    f.readline()
    for line in f:
        qid, ansid, _, rel = line.split(",")
        if int(rel) > 0:
            qrels[qid].add(ansid)
    
    f = open(os.path.join(input_dir, f"question.csv"))
    f.readline()
    with open(os.path.join(output_dir, f"query.{mode}"), 'w') as query_output, \
            open(os.path.join(output_dir, f"qrels.{mode}"), 'w') as qrel_output:
        for line in f:
            qid, query = line.split(",", maxsplit=1)
            query = query.strip()
            if qid in qrels:
                query_output.write(f"{qid}\t{query}\n")
                for ansid in qrels[qid]:
                    qrel_output.write(f"{qid}\t0\t{ansid}\t1\n")    


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    format_corpus()
    format_train()
    format_test_queries("dev")
    format_test_queries("test")