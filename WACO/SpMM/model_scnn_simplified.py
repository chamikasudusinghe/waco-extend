import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME 

class ResNetBase(nn.Module): 
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 32
    PLANES = (16,32,64,64) 

    def __init__(self, in_channels, out_channels, D=2):
        nn.Module.__init__(self)
        self.D = D
        self.inplanes = self.INIT_DIM 

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        # Sparse Matrix Query (SCNN part)
        # self.inplanes has been set in __init__

        # === MODIFICATION 1: Reduce Convolutional Layers ===
        self.layer1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=5, stride=1, dimension=D),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))
        self.layer3 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))
        self.layer4 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))
        # REMOVE self.layer5 through self.layer14 definitions
        # # self.layer5 = nn.Sequential(...)
        # # ...
        # # self.layer14 = nn.Sequential(...)

        # === MODIFICATION 2: glob_pool definition remains, but usage changes in embed_sparse_matrix ===
        self.glob_pool = nn.Sequential(
            ME.MinkowskiGlobalAvgPooling(),
            ME.MinkowskiToFeature())

        # Auxiliary Feature Processing (No change here)
        self.feature = nn.Sequential(
          nn.Linear(3, 64),
          nn.ReLU(),
          nn.Linear(64,32),
        )

        # === MODIFICATION 3: Simplify the Final Embedding MLP ===
        # Original was: nn.Linear(self.INIT_DIM*14+32, 256) -> ReLU -> nn.Linear(256,128)
        # New input size: self.INIT_DIM (from one pooled vector) + 32 (from self.feature) = 64
        self.matrix_embedding = nn.Sequential(
          nn.Linear(self.INIT_DIM + 32, 128), # Adjusted input from 480 to 64
          nn.ReLU(),
          # Removed the second nn.Linear(256, 128) for further simplification
        )

        # --- Super Schedule parts (SpMM specific - UNCHANGED for this SCNN test) ---
        self.isplit = nn.Embedding(17, 32)
        self.ksplit = nn.Embedding(17, 32)
        self.jsplit = nn.Embedding(8, 32)
        self.formati1 = nn.Embedding(2, 32)
        self.formati0 = nn.Embedding(2, 32)
        self.formatk1 = nn.Embedding(2, 32)
        self.formatk0 = nn.Embedding(2, 32)
        self.parchunk = nn.Embedding(9, 32)
        self.order = nn.Linear(36, 32)

        self.schedule_embedding = nn.Sequential(
            nn.Linear(32*9,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )
        # --- End of Super Schedule Parts ---

        # Final Layer (UNCHANGED)
        self.final = nn.Sequential(
            nn.Linear(128+128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        );

    def weight_initialization(self): # Keep or adapt as needed
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear): # Good to initialize linear layers too
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # === MODIFICATION 4: Simplify the embed_sparse_matrix method ===
    def embed_sparse_matrix(self, x1: ME.SparseTensor, x2) :
        # Pass through the remaining 4 layers
        y1_out = self.layer1(x1) # Renamed to avoid clash if original y1, y2... were global
        y2_out = self.layer2(y1_out)
        y3_out = self.layer3(y2_out)
        y4_out = self.layer4(y3_out) # Output of the last SCNN conv layer

        # REMOVE processing of y5 through y14

        # Apply global average pooling *only* to the final output (y4_out)
        y_pooled = self.glob_pool(y4_out) # Shape: Batch x INIT_DIM (e.g., Batch x 32)

        # REMOVE pooling of y1_out, y2_out, y3_out and the large concatenation:
        # # y1_pooled  = self.glob_pool(y1_out)
        # # ...
        # # y_concatenated = torch.cat((y1_pooled, y2_pooled, ..., y14_pooled), dim=1)

        # Process auxiliary features (No change here)
        x2_feat = self.feature(x2[:, :3]) # Shape: Batch x 32

        # Concatenate the single pooled sparse features and auxiliary features
        combined_feat = torch.cat((y_pooled, x2_feat), dim=1) # Shape: Batch x 64

        # Pass through the simplified embedding MLP
        matrix_embedding_output = self.matrix_embedding(combined_feat) # Shape: Batch x 128

        # Optional: L2 normalize the output embedding
        # matrix_embedding_output = F.normalize(matrix_embedding_output, p=2, dim=1)

        return matrix_embedding_output

    # --- embed_super_schedule (UNCHANGED for this SCNN test) ---
    def embed_super_schedule(self, y) :
        # This remains exactly as it was in your original spmm/model.py
        isplit = self.isplit(y[:, 0].long())
        ksplit = self.ksplit(y[:, 1].long())
        jsplit = self.jsplit(y[:, 2].long())
        order = self.order(y[:, 3:39])
        i1f = self.formati1(y[:, 39].long())
        i0f = self.formati0(y[:, 40].long())
        k1f = self.formatk1(y[:, 41].long())
        k0f = self.formatk0(y[:, 42].long())
        pchk = self.parchunk(y[:, 45].long()) # Assuming SpMM indexing
        y_cat = torch.cat((isplit,ksplit,jsplit,order,i1f,i0f,k1f,k0f,pchk), dim=1)
        y_embedded = self.schedule_embedding(y_cat)
        return y_embedded

    # --- forward_after_query (UNCHANGED) ---
    def forward_after_query(self, x, y): # x is matrix_embedding, y is schedule_params
        # This method likely takes the *already embedded matrix feature x*
        # and the raw schedule parameters y, then embeds y and combines.
        # OR it takes embedded matrix and embedded schedule. Let's assume it embeds schedule:
        y_embedded = self.embed_super_schedule(y)

        # Ensure x (matrix_embedding) is expanded if y_embedded has a larger batch size
        if x.shape[0] == 1 and y_embedded.shape[0] > 1:
            x = x.expand(y_embedded.shape[0], -1)
        elif x.shape[0] != y_embedded.shape[0] and y_embedded.shape[0] != 0 : # Avoid error on empty schedule batch
             raise ValueError(f"Batch size mismatch: Matrix embedding ({x.shape[0]}) vs Schedule embedding ({y_embedded.shape[0]})")


        xy = torch.cat((x,y_embedded), dim=1)
        xy = self.final(xy)
        return xy

    # --- forward (Verify logic with original train.py usage) ---
    def forward(self, x1: ME.SparseTensor, x2, y_schedule_params):
        # x1: ME.SparseTensor (matrix)
        # x2: shapes (aux features)
        # y_schedule_params: raw schedule parameters tensor

        matrix_embedding_output = self.embed_sparse_matrix(x1, x2)
        # The original train.py expanded query_feature (matrix_embedding_output) here.
        # This is now handled inside forward_after_query if schedule batch > 1.

        prediction = self.forward_after_query(matrix_embedding_output, y_schedule_params)
        return prediction


class ResNet14(ResNetBase): 
    LAYERS = (1, 1, 1, 1) 
