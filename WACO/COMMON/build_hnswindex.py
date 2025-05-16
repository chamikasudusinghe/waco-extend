import numpy as np
import torch
import torch.nn as nn
import argparse
from model import ResNet14
from Loader.superschedule_loader import TrainingScheduleDataset
from Loader.sparsematrix_loader import SparseMatrixDataset, collate_fn
import MinkowskiEngine as ME
import hnswlib
import time
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['spmm', 'sddmm'], help="Choose mode: 'spmm' or 'sddmm'")
    parser.add_argument('--input', type=str, default="/home/chamika2/waco-extend/total.txt", help="Path to schedule list file")
    parser.add_argument('--output', type=str, default="/home/chamika2/waco-extend/hnswlib/WACO_COSTMODEL", help="Output directory")
    parser.add_argument('--weights-prefix', type=str, default="weight", help="Prefix for saved weight files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    schedules = TrainingScheduleDataset(args.input, mode=args.mode)
    schedule_loader = torch.utils.data.DataLoader(schedules, batch_size=128, shuffle=False, num_workers=0)

    net = ResNet14(in_channels=1, out_channels=1, D=2)
    net = net.to(device)
    net.load_state_dict(torch.load('resnet.pth'))
    net.eval()

    os.makedirs(args.output, exist_ok=True)

    for i, layer in enumerate([net.final[0], net.final[2], net.final[4]]):
        np.savetxt(os.path.join(args.output, f"{args.weights_prefix}{i}.txt"), layer.weight.detach().cpu().numpy().flatten(), fmt='%.6f')
        np.savetxt(os.path.join(args.output, f"bias{i}.txt"), layer.bias.detach().cpu().numpy().flatten(), fmt='%.6f')

    start = time.time()
    names = []
    embeddings = []

    for batch_idx, (data, string) in enumerate(schedule_loader):
        data = data.to(device)
        embedding = net.embed_super_schedule(data)
        embeddings.extend(embedding.detach().cpu().tolist())
        names.extend(string)

    embeddings = np.array(embeddings)
    print("Calculate Embedding Time:", time.time() - start)

    dim = embeddings.shape[1]
    num_elements = embeddings.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=200, M=32)

    start = time.time()
    p.add_items(embeddings, np.arange(num_elements))
    print("Generate HNSW Index Time:", time.time() - start)

    index_path = os.path.join(f"hnsw_schedule_{args.mode}.bin")
    p.save_index(index_path)
    print(f"Saved HNSW index to: {index_path}")
