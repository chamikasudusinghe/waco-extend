import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

# Part 1: The Simplified SCNN Module (Jiangshan's core task)
class SimplifiedSCNN(nn.Module):
    """
    A simplified Sparsity CNN module based on the original ResNetBase,
    designed for extracting features from sparse matrix representations.

    Changes from original:
    - Reduced conv layers from 14 to 4.
    - Applies global average pooling only to the final conv layer output.
    - Simplified the final MLP (`matrix_embedding`) accordingly.
    """
    INIT_DIM = 32 # Keep the original filter count for now, can be tuned later

    def __init__(self, in_channels=1, D=2):
        super().__init__()
        self.D = D
        self.inplanes = self.INIT_DIM

        # --- Convolutional Layers ---
        # Kept Layer 1 (5x5 kernel)
        self.layer1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=5, stride=1, dimension=D),
            ME.MinkowskiReLU(inplace=True),
        )
        # Kept Layers 2-4 (3x3 kernel, stride 2 for downsampling)
        self.layer2 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True)
        )
        # Layers 5-14 are REMOVED for simplification

        # --- Pooling ---
        # Global Average Pooling - applied only ONCE after the last conv layer
        self.glob_pool = nn.Sequential(
            ME.MinkowskiGlobalAvgPooling(),
            ME.MinkowskiToFeature() # Converts sparse tensor to dense tensor
        )

        # --- Auxiliary Feature Processing ---
        # Processes dense features like matrix shape (if provided)
        # Input: Expects x2 to have at least 3 features (e.g., shape)
        # Output: 32-dim feature vector
        self.feature = nn.Sequential(
          nn.Linear(3, 64), # Assumes first 3 elements of x2 are relevant
          nn.ReLU(),
          nn.Linear(64, 32),
        )

        # --- Final Matrix Embedding MLP ---
        # Combines pooled sparse features and auxiliary features.
        # Input size = Output of glob_pool (INIT_DIM) + Output of feature (32)
        # Input size = 32 + 32 = 64
        # Output size = 128 (to match original expected embedding size)
        self.matrix_embedding = nn.Sequential(
          nn.Linear(self.INIT_DIM + 32, 128), # Adjusted input size from 480 to 64
          nn.ReLU(),
          # Removed the second linear layer (256->128) for simplicity
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x1: ME.SparseTensor, x2):
        """
        Forward pass for the SCNN module.
        Args:
            x1 (ME.SparseTensor): The sparse tensor representation of the matrix.
            x2 (torch.Tensor): Auxiliary dense features (e.g., shape). Batch x Features.
        Returns:
            torch.Tensor: A 128-dimensional embedding vector for the matrix.
        """
        # Pass through convolutional layers
        y1 = self.layer1(x1)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3) # Output of the last conv layer

        y_pooled = self.glob_pool(y4) # Shape: Batch x INIT_DIM (Batch x 32)

        # Process auxiliary features
        x2_feat = self.feature(x2[:, :3]) # Shape: Batch x 32

        combined_feat = torch.cat((y_pooled, x2_feat), dim=1) # Shape: Batch x 64

        # Pass through the final embedding MLP
        matrix_embedding = self.matrix_embedding(combined_feat) # Shape: Batch x 128


        return matrix_embedding
