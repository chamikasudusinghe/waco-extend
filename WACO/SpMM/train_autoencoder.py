import torch
import torch.nn as nn
from torch.optim import Adam
from model import ResNet14
from Loader.sparsematrix_loader import SparseMatrixDataset, collate_fn
import MinkowskiEngine as ME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
net = ResNet14(in_channels=1, out_channels=1, D=2).to(device)

# Loss & Optimizer
loss_fn = nn.MSELoss()
optimizer = Adam(net.parameters(), lr=1e-4)

# Data
dataset = SparseMatrixDataset('./TrainingData/train.txt')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

# Training loop
for epoch in range(30):
    net.train()
    total_loss = 0
    for mtx_names, coords, features, shapes in dataloader:
        SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)
        shapes = shapes.to(device)

        optimizer.zero_grad()
        _, reconstructed_shape = net.forward_autoencoder(SparseMatrix, shapes)
        loss = loss_fn(reconstructed_shape, shapes)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: MSE Loss = {total_loss / len(dataloader)}")
