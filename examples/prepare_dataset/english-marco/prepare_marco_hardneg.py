import os
import sys
import json
import gzip
import pickle
import numpy as np
from tqdm import tqdm

hardneg_file = sys.argv[1]
output_path = sys.argv[2]

lengths = []
with open(output_path, 'w') as output_file:
    with gzip.open(hardneg_file, 'rt') as fIn:
        for line in tqdm(fIn):
            data = json.loads(line)
            qid = data['qid']
            pos_pids = data['pos']
            system_negs = set(sum(data['neg'].values(), []))
            system_negs = system_negs - set(pos_pids)
            assert len(system_negs & set(pos_pids)) == 0
            if len(pos_pids) > 0:
                output_file.write(f"{qid}\t{' '.join((str(x) for x in system_negs))}\n")
                lengths.append(len(system_negs))

# print(np.mean(lengths), np.max(lengths), np.min(lengths))
print("Done")