import numpy as np


def read_fvecs(filename, c_contiguous=True):
    print(f"Reading from {filename}.")
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def read_ivecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


if __name__ == '__main__':
    username = 'bianzheng'
    for dataset in ['siftsmall', 'sift', 'deep', 'glove']:
        base_l = read_fvecs(f'/home/{username}/RaBitQ/data/{dataset}/{dataset}_base.fvecs')
        query_l = read_fvecs(f'/home/{username}/RaBitQ/data/{dataset}/{dataset}_query.fvecs')
        groundtruth_l = read_ivecs(f'/home/{username}/RaBitQ/data/{dataset}/{dataset}_groundtruth.ivecs')
        print(
            f"dataset {dataset}, base_l shape {base_l.shape}, query_l shape {query_l.shape}, groundtruth_l shape {groundtruth_l.shape}")
