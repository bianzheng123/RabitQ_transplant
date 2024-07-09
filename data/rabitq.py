import numpy as np
import struct
import time
import os
from utils.io import *
from tqdm import tqdm


def Orthogonal(D):
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)
    return Q


def GenerateBinaryCode(X, P):
    XP = np.dot(X, P)
    binary_XP = (XP > 0)
    X0 = np.sum(XP * (2 * binary_XP - 1) / D ** 0.5, axis=1, keepdims=True) / np.linalg.norm(XP, axis=1, keepdims=True)
    return binary_XP, X0


def GenerateIndex(P, X_pad, centroids_pad, cluster_id):
    XP = np.dot(X_pad, P)
    CP = np.dot(centroids_pad, P)
    XP = XP - CP[cluster_id]
    bin_XP = (XP > 0)

    # The inner product between the data vector and the quantized data vector, i.e., <\bar o, o>.
    x0 = np.sum(XP[:, :B] * (2 * bin_XP[:, :B] - 1) / B ** 0.5, axis=1, keepdims=True) / np.linalg.norm(XP, axis=1,
                                                                                                        keepdims=True)

    # To remove illy defined x0
    # np.linalg.norm(XP, axis=1, keepdims=True) = 0 indicates that its estimated distance based on our method has no error.
    # Thus, it should be good to set x0 as any finite non-zero number.
    x0[~np.isfinite(x0)] = 0.8

    bin_XP = bin_XP[:, :B].flatten()
    uint64_XP = np.packbits(bin_XP.reshape(-1, 8, 8)[:, ::-1]).view(np.uint64)
    uint64_XP = uint64_XP.reshape(-1, B >> 6)
    return CP, uint64_XP, x0


if __name__ == "__main__":
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
        # path
        data_path = f'/home/{username}/RaBitQ/data/{dataset}'
        index_path = f'/home/{username}/RaBitQ/index/{dataset}'
        base_data_filename = os.path.join(data_path, f'{dataset}_base.fvecs')

        C = 4096
        centroids_path = os.path.join(index_path, f'{dataset}_centroid_{C}.fvecs')
        dist_to_centroid_path = os.path.join(index_path, f'{dataset}_dist_to_centroid_{C}.fvecs')
        cluster_id_path = os.path.join(index_path, f'{dataset}_cluster_id_{C}.ivecs')

        X = read_fvecs(base_data_filename)
        centroids = read_fvecs(centroids_path)
        cluster_id = read_ivecs(cluster_id_path)

        D = X.shape[1]
        B = (D + 63) // 64 * 64
        MAX_BD = max(D, B)

        projection_path = os.path.join(index_path, f'P_C{C}_B{B}.fvecs')
        randomized_centroid_path = os.path.join(index_path, f'RandCentroid_C{C}_B{B}.fvecs')
        RN_path = os.path.join(index_path, f'RandNet_C{C}_B{B}.Ivecs')
        x0_path = os.path.join(index_path, f'x0_C{C}_B{B}.fvecs')

        X_pad = np.pad(X, ((0, 0), (0, MAX_BD - D)), 'constant')
        centroids_pad = np.pad(centroids, ((0, 0), (0, MAX_BD - D)), 'constant')
        # np.random.seed(0)
        cluster_id = np.squeeze(cluster_id)

        # The inverse of an orthogonal matrix equals to its transpose.
        P = Orthogonal(MAX_BD)
        P = P.T

        CP, uint64_XP, x0 = GenerateIndex(P, X_pad, centroids_pad, cluster_id)

        # Output
        print(f"CP shape {CP.shape}, filename {randomized_centroid_path}")
        print(f"uint64_XP shape {uint64_XP.shape}, filename {RN_path}")
        print(f"x0 shape {x0.shape}, filename {x0_path}")
        print(f"P shape {P.shape}, filename {projection_path}")
        os.system(f'rm {randomized_centroid_path}')
        os.system(f'rm {RN_path}')
        os.system(f'rm {x0_path}')
        os.system(f'rm {projection_path}')
        to_fvecs(randomized_centroid_path, CP)
        to_Ivecs(RN_path, uint64_XP)
        to_fvecs(x0_path, x0)
        to_fvecs(projection_path, P)

        # --------------------------------------------------------------------------
        # generate the identity matrix

        os.makedirs(os.path.join(index_path, 'no_rotation'), exist_ok=True)
        projection_path = os.path.join(index_path, 'no_rotation', f'P_C{C}_B{B}.fvecs')
        randomized_centroid_path = os.path.join(index_path, 'no_rotation', f'RandCentroid_C{C}_B{B}.fvecs')
        RN_path = os.path.join(index_path, 'no_rotation', f'RandNet_C{C}_B{B}.Ivecs')
        x0_path = os.path.join(index_path, 'no_rotation', f'x0_C{C}_B{B}.fvecs')

        # The inverse of an orthogonal matrix equals to its transpose.
        P = np.identity(MAX_BD)
        P = P.T

        CP, uint64_XP, x0 = GenerateIndex(P, X_pad, centroids_pad, cluster_id)

        # Output
        print(f"no_rotation CP shape {CP.shape}, filename {randomized_centroid_path}")
        print(f"no_rotation uint64_XP shape {uint64_XP.shape}, filename {RN_path}")
        print(f"no_rotation x0 shape {x0.shape}, filename {x0_path}")
        print(f"no_rotation P shape {P.shape}, filename {projection_path}")
        to_fvecs(randomized_centroid_path, CP)
        to_Ivecs(RN_path, uint64_XP)
        to_fvecs(x0_path, x0)
        to_fvecs(projection_path, P)
