import numpy as np
import torch
import torch.nn as nn
import argparse
import os
from model import ResNet14
from Loader.superschedule_loader import TrainingScheduleDataset
from Loader.sparsematrix_loader import SparseMatrixDataset, collate_fn
import MinkowskiEngine as ME
import hnswlib
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['spmm', 'sddmm'], help="Choose mode: 'spmm' or 'sddmm'")
    parser.add_argument('--index-dir', type=str, default="./", help="Directory where the HNSW index is stored")
    parser.add_argument('--test-file', type=str, default="/home/chamika2/waco-extend/test.txt", help="Sparse matrix test input file")
    parser.add_argument('--schedule-file', type=str, default="/home/chamika2/waco-extend/total.txt", help="Schedule input file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    schedules = TrainingScheduleDataset(args.schedule_file, mode=args.mode)
    schedule_loader = torch.utils.data.DataLoader(schedules, batch_size=128, shuffle=False, num_workers=0)

    net = ResNet14(in_channels=1, out_channels=1, D=2)
    net = net.to(device)
    net.load_state_dict(torch.load('resnet.pth'))
    net.eval()

    names = []
    for _, (_, string) in enumerate(schedule_loader):
        names.extend(string)

    dim = 128
    num_elements = len(schedules)

    index_path = os.path.join(args.index_dir, f"hnsw_schedule_{args.mode}.bin")
    p = hnswlib.Index(space='ip', dim=dim)
    p.load_index(index_path, max_elements=num_elements)
    p.set_ef(200)

    SparseMatrix_Dataset = SparseMatrixDataset(args.test_file)
    test_loader = torch.utils.data.DataLoader(SparseMatrix_Dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)

    os.makedirs("./topk", exist_ok=True)

    for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(test_loader):
        torch.cuda.empty_cache()
        SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)
        shapes = shapes.to(device)

        query = net.embed_sparse_matrix(SparseMatrix, shapes)
        labels, distances = p.knn_query(query.cpu().detach().numpy()[0], k=5)

        with open(f"./topk/{args.mode}/{mtx_names[0]}.txt", 'w') as f:
            f.write('\n'.join(list(np.array(names)[labels[0]])))
