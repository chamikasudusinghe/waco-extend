import torch
import torch.nn as nn
import numpy as np
import itertools
import os
from itertools import permutations, product

class SuperScheduleDataset(torch.utils.data.Dataset):
    def __init__(self, name, mode='spmm'):
        if mode not in ['spmm', 'sddmm']:
            raise ValueError("mode must be 'spmm' or 'sddmm'")
        
        with open(f"/home/chamika2/waco-extend/dataset/{mode}/{name}.txt") as f:
            lines = [line.split() for line in f.read().splitlines()]

        split_ = [1 << p for p in range(17)]
        format_ = [0, 1]  # (C,U)
        parnum_ = [48]
        p_type = 0

        if mode == 'spmm':
            index_ = ['i1', 'i0', 'k1', 'k0', 'j1', 'j0']
            parchunk_ = [1 << p for p in range(9)]
            p_type = 0
        else:  # sddmm
            index_ = ['i1', 'i0', 'j1', 'j0', 'k1', 'k0']
            parchunk_ = [1 << p for p in range(6)]
            p_type = 1

        schedules, runtimes = [], []

        for line in lines:
            dims = list(map(int, line[0:3]))
            dim_indices = [split_.index(d) for d in dims]

            order = line[3:9]
            perm = np.zeros((len(index_), len(index_)))
            for i, idx in enumerate(order):
                perm[index_.index(idx), i] = 1
            perm = perm.flatten()

            format_indices = [format_.index(int(line[i])) for i in range(9, 13)]
            p1 = index_.index(line[13])
            p2 = parnum_.index(int(line[14]))
            p3 = parchunk_.index(int(line[15]))

            concat = np.hstack(np.array(dim_indices + [perm] + format_indices + [p1, p2, p3, p_type], dtype=object))
            runtime = float(line[-1])

            if runtime < 1000:
                schedules.append(concat)
                runtimes.append([runtime])

        self.schedules = torch.tensor(np.stack(schedules), dtype=torch.float32)
        self.runtimes = torch.tensor(np.stack(runtimes), dtype=torch.float32) / 1000.0

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, idx):
        return self.schedules[idx], self.runtimes[idx]


class TrainingScheduleDataset(torch.utils.data.Dataset):
    def __init__(self, filename, mode='spmm'):
        if mode not in ['spmm', 'sddmm']:
            raise ValueError("mode must be 'spmm' or 'sddmm'")

        split_ = [1 << p for p in range(17)]
        format_ = [0, 1]
        parnum_ = [48]
        p_type = 0

        if mode == 'spmm':
            index_ = ['i1', 'i0', 'k1', 'k0', 'j1', 'j0']
            parchunk_ = [1 << p for p in range(9)]
            p_type = 0
        else:
            index_ = ['i1', 'i0', 'j1', 'j0', 'k1', 'k0']
            parchunk_ = [1 << p for p in range(6)]
            p_type = 1

        schedules = []
        schedules_str = []
        uniqstr = set()

        with open(filename) as f:
            names = f.read().splitlines()

        for name in names:
            with open(f"/home/chamika2/waco-extend/dataset/{mode}/{name}.txt") as f:
                lines = [line.split() for line in f.read().splitlines()]

            for line in lines:
                line_key = " ".join(line[:-2])
                if line_key in uniqstr:
                    continue
                uniqstr.add(line_key)

                dims = list(map(int, line[0:3]))
                dim_indices = [split_.index(d) for d in dims]

                order = line[3:9]
                perm = np.zeros((len(index_), len(index_)))
                for i, idx in enumerate(order):
                    perm[index_.index(idx), i] = 1
                perm = perm.flatten()

                format_indices = [format_.index(int(line[i])) for i in range(9, 13)]
                p1 = index_.index(line[13])
                p2 = parnum_.index(int(line[14]))
                p3 = parchunk_.index(int(line[15]))

                concat = np.hstack(np.array(dim_indices + [perm] + format_indices + [p1, p2, p3, p_type], dtype=object))
                schedules.append(concat)
                schedules_str.append(line_key)

        self.schedules = torch.tensor(np.stack(schedules), dtype=torch.float32)
        self.schedules_str = schedules_str

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, idx):
        return self.schedules[idx], self.schedules_str[idx]
