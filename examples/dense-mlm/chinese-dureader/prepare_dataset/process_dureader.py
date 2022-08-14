import os
import sys
import json
import hashlib
from tqdm import tqdm


def get_text_id(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def concat_title_body(title, body, sep_token):
    title = title.strip()
    if len(title)> 0:
        if title[-1] in ",，!.?。！？":
            text = title + " " + body
        else:
            text = title + sep_token + body
    else:
        text = body
    return text


def main_rel_passage():
    input_root = sys.argv[1]
    output_dir = sys.argv[2]
    assert os.path.exists(input_root), input_root
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "corpus.tsv"), 'w') as output_corpus:
        corpus_id_set = set()
        for mode in tqdm(["dev", "train"], desc="dev or train"):

            with open(os.path.join(output_dir, f"query.{mode}"), 'w') as output_query, \
                    open(os.path.join(output_dir, f"qrels.{mode}"), 'w') as output_qrel:
                query_id_set, qrel_set = set(), set()

                for topic in tqdm(["search", "zhidao"], desc="[search, zhidao]"):
                    for line in tqdm(open(os.path.join(input_root, f"{mode}set", f"{topic}.{mode}.json")), desc="processing"):
                        data = json.loads(line)
                        query = data['question'].strip().replace("\n", " ").replace("\t", " ").replace("\r", " ")
                        qid = get_text_id(query)
                        
                        if qid not in query_id_set:
                            output_query.write(f"{qid}\t{query}\n")
                            query_id_set.add(qid)

                        for doc_idx, doc_data in enumerate(data['documents']):
                            title = doc_data['title'].strip().replace("\n", " ").replace("\t", " ").replace("\r", " ")
                            for para_idx, para in enumerate(doc_data['paragraphs']):
                                para = para.strip().replace("\n", " ").replace("\t", " ").replace("\r", " ")
                                concat_text = concat_title_body(title, para, sep_token=" ")

                                pid = get_text_id(concat_text)
                                if doc_idx in data['answer_docs'] and doc_data['most_related_para'] == para_idx:
                                    if (qid, pid) not in qrel_set:
                                        output_qrel.write(f"{qid}\t0\t{pid}\t1\n")
                                        qrel_set.add((qid, pid))
                                    
                                    if pid not in corpus_id_set:
                                        output_corpus.write(f"{pid}\t{concat_text}\n")
                                        corpus_id_set.add(pid)

 
if __name__ == "__main__":
    main_rel_passage()