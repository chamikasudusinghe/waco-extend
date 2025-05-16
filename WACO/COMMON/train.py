import os
import random 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import matplotlib
import matplotlib.pyplot as plt 
import sys
from model import ResNet14
from Loader.superschedule_loader import SuperScheduleDataset
from Loader.sparsematrix_loader import SparseMatrixDataset, collate_fn
import MinkowskiEngine as ME

def get_schedule_mode(index):
    return "spmm" if index % 2 == 0 else "sddmm"

if __name__ == "__main__":
    f = open("trainlog.txt",'a')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = ResNet14(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
    net = net.to(device)
    #net.load_state_dict(torch.load('./resnet.pth'))
  
    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(net.parameters(), lr=1e-4)    
    
    SparseMatrix_Dataset = SparseMatrixDataset('/home/chamika2/waco-extend/train.txt')
    train_SparseMatrix = torch.utils.data.DataLoader(SparseMatrix_Dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    SparseMatrix_Dataset_Valid = SparseMatrixDataset('/home/chamika2/waco-extend/validation.txt')
    valid_SparseMatrix = torch.utils.data.DataLoader(SparseMatrix_Dataset_Valid, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    for epoch in range(10) :
        net.train()
        train_loss = 0
        train_loss_cnt = 0 
        for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(train_SparseMatrix):
            torch.cuda.empty_cache()
            torch.save(net.state_dict(), "resnet.pth")

            mode = get_schedule_mode(sparse_batchidx)
            SuperSchedule_Dataset = SuperScheduleDataset(mtx_names[0], mode=mode)
            train_SuperSchedule = torch.utils.data.DataLoader(SuperSchedule_Dataset, batch_size=32, shuffle=True, num_workers=0)

            shapes = shapes.to(device)
            SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)

            for schedule_batchidx, (schedule, runtime) in enumerate(train_SuperSchedule):
                if schedule.shape[0] < 2:
                    break

                schedule, runtime = schedule.to(device), runtime.to(device)
                optimizer.zero_grad()

                query_feature = net.embed_sparse_matrix(SparseMatrix, shapes)
                query_feature = query_feature.expand((schedule.shape[0], query_feature.shape[1]))
                predict = net.forward_after_query(query_feature, schedule)

                iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
                pred1, pred2 = predict[iu[0]], predict[iu[1]]
                true1, true2 = runtime[iu[0]], runtime[iu[1]]
                sign = (true1 - true2).sign()
                loss = criterion(pred1, pred2, sign)
                train_loss += loss.item()
                train_loss_cnt += 1

                loss.backward()
                optimizer.step()

                if sparse_batchidx % 100 == 0 and schedule_batchidx == 0:
                    print(f"[Train] Epoch {epoch}, MTX: {mtx_names[0]}, Mode: {mode}, Shapes: {shapes}, Loss: {loss.item():.4f}")
                    print("\tPredict:", predict.detach()[:5, 0])
                    print("\tGT     :", runtime.detach()[:5, 0])
                    print("\tQuery  :", query_feature.detach()[0, :5])
                break  # only one schedule batch per matrix
      #Validation
        net.eval()
        valid_loss = 0
        valid_loss_cnt = 0

        with torch.no_grad():
            for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(valid_SparseMatrix):
                torch.cuda.empty_cache()

                mode = get_schedule_mode(sparse_batchidx)
                SuperSchedule_Dataset = SuperScheduleDataset(mtx_names[0], mode=mode)
                valid_SuperSchedule = torch.utils.data.DataLoader(SuperSchedule_Dataset, batch_size=32, shuffle=True, num_workers=0)
                shapes = shapes.to(device)
                SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)

                for schedule_batchidx, (schedule, runtime) in enumerate(valid_SuperSchedule):
                    if schedule.shape[0] < 6:
                        break

                    schedule, runtime = schedule.to(device), runtime.to(device)
                    query_feature = net.embed_sparse_matrix(SparseMatrix, shapes)
                    query_feature = query_feature.expand((schedule.shape[0], query_feature.shape[1]))
                    predict = net.forward_after_query(query_feature, schedule)

                    iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
                    pred1, pred2 = predict[iu[0]], predict[iu[1]]
                    true1, true2 = runtime[iu[0]], runtime[iu[1]]
                    sign = (true1 - true2).sign()
                    loss = criterion(pred1, pred2, sign)

                    valid_loss += loss.item()
                    valid_loss_cnt += 1

                    if sparse_batchidx % 100 == 0 and schedule_batchidx == 0:
                        print(f"[Valid] Epoch {epoch}, MTX: {mtx_names[0]}, Mode: {mode}, Shapes: {shapes}, Loss: {loss.item():.4f}")
                        print("\tValidPredict:", predict.detach()[:5, 0])
                        print("\tValidGT     :", runtime.detach()[:5, 0])
                        print("\tValidQuery  :", query_feature.detach()[0, :5])
                    break  # only one schedule batch per matrix

        train_avg = train_loss / train_loss_cnt if train_loss_cnt else 0
        valid_avg = valid_loss / valid_loss_cnt if valid_loss_cnt else 0

        log = f"--- Epoch {epoch} : Train {train_avg:.6f} Valid {valid_avg:.6f} ---"
        print(log)
        f.write(log + "\n")
        f.flush()
