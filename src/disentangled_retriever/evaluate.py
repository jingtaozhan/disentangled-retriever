import os
import sys
import json
import logging
import pytrec_eval
from typing import Dict, Union
from collections import defaultdict


logger = logging.getLogger(__name__)


def truncate_run(run: Dict[str, Dict[str, float]], topk: int):
    new_run = dict()
    for qid, pid2scores in run.items():
        rank_lst = sorted(pid2scores.items(), key=lambda x: x[1], reverse=True)
        new_run[qid] = dict(rank_lst[:topk])
    return new_run


def pytrec_evaluate(
        qrel: Union[str, Dict[str, Dict[str, int]]], 
        run: Union[str, Dict[str, float]],
        k_values =(1, 3, 5, 10, 100),
        mrr_k_values = (10, 100),
        recall_k_values = (10, 50, 100, 200, 500, 1000),
        relevance_level = 1,
        ):
    ndcg, map, recall, precision, mrr = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in recall_k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    if isinstance(qrel, str):
        with open(qrel, 'r') as f_qrel:
            qrel = pytrec_eval.parse_qrel(f_qrel)
    if isinstance(run, str):
        with open(run, 'r') as f_run:
            run = pytrec_eval.parse_run(f_run)

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {map_string, ndcg_string, recall_string, precision_string}, relevance_level=relevance_level)
    query_scores = evaluator.evaluate(run)
    
    for query_id in query_scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += query_scores[query_id]["ndcg_cut_" + str(k)]
            map[f"MAP@{k}"] += query_scores[query_id]["map_cut_" + str(k)]
            precision[f"P@{k}"] += query_scores[query_id]["P_"+ str(k)]
        for k in recall_k_values:
            recall[f"Recall@{k}"] += query_scores[query_id]["recall_" + str(k)]
    
    if len(query_scores) < len(qrel):
        missing_qids = qrel.keys() - query_scores.keys()
        logger.warning(f"Missing results for {len(missing_qids)} queries!")
        for query_id in missing_qids:
            query_scores[query_id] = dict()
            for k in k_values:
                query_scores[query_id]["ndcg_cut_" + str(k)] = 0
                query_scores[query_id]["map_cut_" + str(k)] = 0
                query_scores[query_id]["P_"+ str(k)] = 0
            for k in recall_k_values:
                query_scores[query_id]["recall_" + str(k)] = 0

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(query_scores), 5)
        map[f"MAP@{k}"] = round(map[f"MAP@{k}"]/len(query_scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(query_scores), 5)
    for k in recall_k_values:
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(query_scores), 5)

    mrr_evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {"recip_rank"}, relevance_level=relevance_level)
    for mrr_cut in mrr_k_values:
        mrr_query_scores = mrr_evaluator.evaluate(truncate_run(run, mrr_cut))
        for query_id in mrr_query_scores.keys():
            s = mrr_query_scores[query_id]["recip_rank"]
            mrr[f"MRR@{mrr_cut}"] += s
            query_scores[query_id][f"recip_rank_{mrr_cut}"] = s
        mrr[f"MRR@{mrr_cut}"] = round(mrr[f"MRR@{mrr_cut}"]/len(mrr_query_scores), 5)

    ndcg, map, recall, precision, mrr = dict(ndcg), dict(map), dict(recall), dict(precision), dict(mrr)
    metric_scores = {
        "ndcg": ndcg,
        "map": map,
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
        "perquery": query_scores
    }
    return dict(metric_scores)


if __name__ == "__main__":
    assert len(sys.argv) == 4
    qrel_path = sys.argv[1]
    run_path = sys.argv[2]
    print(qrel_path)
    assert os.path.exists(run_path) and os.path.exists(qrel_path)
    output_path = sys.argv[3]
    assert not os.path.exists(output_path)
    metric_scores = pytrec_evaluate(
        qrel_path, 
        run_path
    )
    for k in metric_scores.keys():
        if k != "perquery":
            print(metric_scores[k])
    json.dump(metric_scores, open(output_path, 'w'), indent=1)
    