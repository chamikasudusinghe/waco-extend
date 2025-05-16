import torch
import torch.nn as nn
from torch.optim import Adam
from model import ResNet14
from Loader.sparsematrix_loader_autoencoder import SparseMatrixDataset, collate_fn
import MinkowskiEngine as ME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
net = ResNet14(in_channels=1, out_channels=1, D=2).to(device)

# Loss & Optimizer
loss_fn = nn.MSELoss()
optimizer = Adam(net.parameters(), lr=1e-4)

# Data
dataset = SparseMatrixDataset('/home/chamika2/waco-extend/autoencoder.txt')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
print("Data loading completed")

# Early stopping settings
best_loss = float('inf')
patience = 5
counter = 0

# Training loop
for epoch in range(100):  # Allow more epochs for early stopping to work
    net.train()
    total_loss = 0
    for mtx_names, coords, features, shapes, targets in dataloader:
        SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)
        shapes, targets = shapes.to(device), targets.to(device)

        optimizer.zero_grad()
        _, reconstructed_matrix, reconstructed_shape = net.forward_autoencoder(SparseMatrix, shapes)

        loss_matrix = loss_fn(reconstructed_matrix, targets)
        loss_shape = loss_fn(reconstructed_shape, shapes)
        loss = loss_matrix + loss_shape

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}: Total Loss = {avg_loss:.6f}")
    torch.save(net.state_dict(), "scnn_weights_epoch.pth")

    # Early stopping check
    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
        torch.save(net.state_dict(), "scnn_weights.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break
