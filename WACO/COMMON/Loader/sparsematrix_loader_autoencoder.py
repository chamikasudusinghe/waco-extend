import torch
import numpy as np
import MinkowskiEngine as ME
import os
from scipy.sparse import coo_matrix
import cv2
import multiprocessing

def from_csr(filename):
    csr = np.fromfile("/home/chamika2/waco-extend/dataset/" + filename + ".csr", dtype='<i4')
    num_row, num_col, nnz = csr[0], csr[1], csr[2]
    coo = np.zeros((nnz, 2), dtype=int)
    coo[:, 1] = csr[3 + num_row + 1:]
    bins = np.array(csr[4:num_row + 4]) - np.array(csr[3:num_row + 3])
    coo[:, 0] = np.repeat(range(num_row), bins)
    return num_row, num_col, nnz, coo

def collate_fn(list_data):
    coords_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["label"] for d in list_data],
    )

    mtxnames_batch = [d["mtxname"] for d in list_data]
    shapes_batch = torch.stack([d["shape"] for d in list_data])
    targets_batch = torch.stack([d["target_matrix"] for d in list_data])

    return mtxnames_batch, coords_batch, features_batch, shapes_batch, targets_batch

def process_matrix(args):
    filename, resolution, standardize = args

    num_row, num_col, nnz, coo = from_csr(filename)
    dense_matrix = coo_matrix((np.ones(len(coo)), (coo[:, 0], coo[:, 1])), shape=(num_row, num_col)).toarray()
    downsampled = cv2.resize(dense_matrix, (resolution, resolution), interpolation=cv2.INTER_AREA)
    target_matrix = torch.tensor(downsampled, dtype=torch.float32).unsqueeze(0)

    coordinates = torch.from_numpy(coo).to(torch.int32)
    features = torch.ones((len(coo), 1), dtype=torch.float32)
    label = torch.tensor([[0]], dtype=torch.float32)

    shape = torch.tensor([
        (num_row - standardize["mean_rows"]) / standardize["std_rows"],
        (num_col - standardize["mean_cols"]) / standardize["std_cols"],
        (nnz - standardize["mean_nnzs"]) / standardize["std_nnzs"]
    ], dtype=torch.float32)

    return {
        "mtxname": filename,
        "coordinates": coordinates,
        "features": features,
        "label": label,
        "shape": shape,
        "target_matrix": target_matrix,
    }

class SparseMatrixDataset(torch.utils.data.Dataset):
    def __init__(self, filename, resolution=32, num_workers=None):
        with open(filename) as f:
            self.names = f.read().splitlines()

        self.resolution = resolution
        self.standardize = self.compute_standardization()

        # Preload in parallel
        self.data = self.preload_all_matrices(num_workers=num_workers)
        print("Preloading completed")

    def compute_standardization(self):
        stats = {"rows": [], "cols": [], "nnzs": []}
        with open("/home/chamika2/waco-extend/train.txt") as f:
            for filename in f.read().splitlines():
                csr = np.fromfile("/home/chamika2/waco-extend/dataset/" + filename + ".csr", count=3, dtype='<i4')
                stats["rows"].append(csr[0])
                stats["cols"].append(csr[1])
                stats["nnzs"].append(csr[2])

        return {
            "mean_rows": np.mean(stats["rows"]),
            "mean_cols": np.mean(stats["cols"]),
            "mean_nnzs": np.mean(stats["nnzs"]),
            "std_rows": np.std(stats["rows"]),
            "std_cols": np.std(stats["cols"]),
            "std_nnzs": np.std(stats["nnzs"]),
        }

    def preload_all_matrices(self, num_workers=None):
        print(f"[INFO] Preloading {len(self.names)} matrices with {num_workers or os.cpu_count()} workers...")
        with multiprocessing.Pool(processes=num_workers) as pool:
            args = [(name, self.resolution, self.standardize) for name in self.names]
            data = pool.map(process_matrix, args)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]