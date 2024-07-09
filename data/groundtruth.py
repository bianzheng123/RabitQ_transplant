import os.path

import faiss
import numpy as np
from utils import io


def ip_gnd(base, query, k):
    base_dim = base.shape[1]
    index = faiss.IndexFlatIP(base_dim)
    index.add(base)
    gnd_distance, gnd_idx = index.search(query, k)
    return gnd_idx, gnd_distance


def l2_gnd(base, query, k):
    base_dim = base.shape[1]
    index = faiss.IndexFlatL2(base_dim)
    index.add(base)
    gnd_distance, gnd_idx = index.search(query, k)
    print("search")
    return gnd_idx, gnd_distance


if __name__ == '__main__':
    dataset = 'glove'
    username = 'bianzheng'
    data_path = f'/home/{username}/RaBitQ/data/{dataset}'
    k = 100

    base_filename = os.path.join(data_path, f'{dataset}_base.fvecs')
    query_filename = os.path.join(data_path, f'{dataset}_query.fvecs')
    base_l = io.read_fvecs(base_filename)
    query_l = io.read_fvecs(query_filename)
    gnd_idx, gnd_distance = l2_gnd(base=base_l, query=query_l, k=100)
    io.to_ivecs(os.path.join(data_path, f'{dataset}_groundtruth.ivecs'), gnd_idx)
    print(
        f"dataset {dataset}, base_l shape {base_l.shape}, query_l shape {query_l.shape}, groundtruth_l shape {gnd_idx.shape}")
