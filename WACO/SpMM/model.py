import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 32 
    PLANES = (16,32,64,64)

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        # Sparse Matrix Query 
        self.inplanes = self.INIT_DIM

        self.feature = nn.Sequential(
          nn.Linear(3, 64),
          nn.ReLU(),
          nn.Linear(64,32),
        )
        
        self.matrix_embedding = nn.Sequential(
          nn.Linear(self.INIT_DIM*14+32, 256),
          nn.ReLU(),
          nn.Linear(256,128),
        )

        # Super Schedule
        self.isplit = nn.Embedding(17, 32)
        self.ksplit = nn.Embedding(17, 32)
        self.jsplit = nn.Embedding(8, 32)
        self.formati1 = nn.Embedding(2, 32)
        self.formati0 = nn.Embedding(2, 32)
        self.formatk1 = nn.Embedding(2, 32)
        self.formatk0 = nn.Embedding(2, 32)
        self.parchunk = nn.Embedding(9, 32) # For OpenTuner
        self.order = nn.Linear(36, 32) #6x6 Permutation

        self.schedule_embedding = nn.Sequential(
            nn.Linear(32*9,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )
        
        # Final Layer
        self.final = nn.Sequential(
            nn.Linear(128+128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        );

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def embed_sparse_matrix_shape(self, x2) :
        # Sparse Matrix
        y = torch.zeros((x2.size(0), 448), device=x2.device)

        x2 = self.feature(x2[:, :3])
        x1x2 = torch.cat((y,x2), dim=1)
        x1x2 = self.matrix_embedding(x1x2)
        
        #x1x2 = F.normalize(x1x2)

        return x1x2

    def embed_super_schedule(self, y) :
        # Super Schedule
        isplit = self.isplit(y[:, 0].long())
        ksplit = self.ksplit(y[:, 1].long())
        jsplit = self.jsplit(y[:, 2].long())
        order = self.order(y[:, 3:39])
        i1f = self.formati1(y[:, 39].long())
        i0f = self.formati0(y[:, 40].long())
        k1f = self.formatk1(y[:, 41].long())
        k0f = self.formatk0(y[:, 42].long())
        #pidx = self.paridx(y[:, 43].long())
        #pnum = self.parnum(y[:, 44].long())
        pchk = self.parchunk(y[:, 45].long())
        y = torch.cat((isplit,ksplit,jsplit,order,i1f,i0f,k1f,k0f,pchk), dim=1)
        y = self.schedule_embedding(y)

        #y = F.normalize(y)
        return y

    def forward_after_query(self, x, y):
        y = self.embed_super_schedule(y)
        xy = torch.cat((x,y), dim=1)
        xy = self.final(xy)
        return xy

class ResNet14(ResNetBase):
    LAYERS = (1, 1, 1, 1)

