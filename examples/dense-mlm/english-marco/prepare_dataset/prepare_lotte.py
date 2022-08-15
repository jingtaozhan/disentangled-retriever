import os
import sys
import ir_datasets
from tqdm import tqdm

def main(output_root):
    os.makedirs(output_root, exist_ok=True)
    output_root = "./data/datasets/lotte"

    for topic in tqdm(['lifestyle', 'recreation', 'science', 'technology', 'writing']):
        mode = 'test'
        q_type = 'forum'
        output_dir = os.path.join(output_root, topic, mode)
        os.makedirs(output_dir, exist_ok=True)

        # write corpus
        with open(os.path.join(output_dir, 'corpus.tsv'), 'w') as output_corpus:
            dataset = ir_datasets.load(f"lotte/{topic}/{mode}")
            for doc in dataset.docs_iter():
                docid, content = doc.doc_id, doc.text
                output_corpus.write(f"{docid}\t{content.strip()}\n")
        
        # write query
        with open(os.path.join(output_dir, f'query.{q_type}'), 'w') as output_query:
            dataset = ir_datasets.load(f"lotte/{topic}/{mode}/{q_type}")
            for query in dataset.queries_iter():
                qid, content = query.query_id, query.text
                output_query.write(f"{qid}\t{content.strip()}\n")
        
        # write qrels
        with open(os.path.join(output_dir, f'qrels.{q_type}'), 'w') as output_qrel:
            dataset = ir_datasets.load(f"lotte/{topic}/{mode}/{q_type}")
            for qrel in dataset.qrels_iter():
                qid, docid, relevance = qrel.query_id, qrel.doc_id, qrel.relevance
                output_qrel.write(f"{qid}\t0\t{docid}\t{relevance}\n")
            

if __name__ == "__main__":
    main(sys.argv[1]) 
            