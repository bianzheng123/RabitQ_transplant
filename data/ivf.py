import numpy as np
import faiss
import struct
import os
from utils.io import *

np.random.seed(0)

def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


if __name__ == '__main__':
    config_l = {
        'local': {
            'username': 'bianzheng',
            # 'dataset_l': ['siftsmall', 'sift', 'deep', 'glove']
            'dataset_l': ['siftsmall']
        }
    }
    host_name = 'local'
    config = config_l[host_name]
    username = config['username']
    dataset_l = config['dataset_l']

    for dataset in dataset_l:
        data_path = f'/home/{username}/RaBitQ/data/{dataset}'
        index_path = f'/home/{username}/RaBitQ/index/{dataset}'

        print(f"Clustering - {dataset}")
        # path
        base_data_filename = os.path.join(data_path, f'{dataset}_base.fvecs')
        X = read_fvecs(base_data_filename)
        D = X.shape[1]
        K = 4096
        centroids_path = os.path.join(index_path, f'{dataset}_centroid_{K}.fvecs')
        dist_to_centroid_path = os.path.join(index_path, f'{dataset}_dist_to_centroid_{K}.fvecs')
        cluster_id_path = os.path.join(index_path, f'{dataset}_cluster_id_{K}.ivecs')

        # cluster data vectors
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        dist_to_centroid, cluster_id = index.quantizer.search(X, 1)
        dist_to_centroid = dist_to_centroid ** 0.5
        print("X shape", X.shape)
        print("dimensionality", D)
        print("centroid shape", centroids.shape)
        print("dist_to_centroid shape", dist_to_centroid.shape)
        print("cluster_id shape", cluster_id.shape)

        delete_file_if_exist(index_path)
        os.makedirs(index_path, exist_ok=False)

        to_fvecs(dist_to_centroid_path, dist_to_centroid)
        to_ivecs(cluster_id_path, cluster_id)
        to_fvecs(centroids_path, centroids)
