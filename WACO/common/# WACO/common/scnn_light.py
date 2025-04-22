# WACO/common/scnn_light.py
import torch, torch.nn as nn, MinkowskiEngine as ME

class SCNNLight(nn.Module):
    def __init__(self, in_ch=1, sched_dim=128, init_dim=16, D=3, pool_layers=6):
        super().__init__()
        ch = init_dim
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_ch, ch, kernel_size=5, dimension=D),
            ME.MinkowskiReLU(inplace=True),
        )
        # use fewer conv–pool stages
        self.stages = nn.ModuleList([
            nn.Sequential(
                ME.MinkowskiConvolution(ch, ch, kernel_size=3, stride=2, dimension=D),
                ME.MinkowskiReLU(inplace=True),
            ) for _ in range(pool_layers)
        ])
        self.glob_pool = nn.Sequential(
            ME.MinkowskiGlobalAvgPooling(),
            ME.MinkowskiToFeature()
        )
        # tiny dense head for matrix meta‑features (row,col,nnz)
        self.meta = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 16)
        )
        self.matrix_embed = nn.Sequential(
            nn.Linear(ch * pool_layers + 16, 128),
            nn.ReLU(),
        )

        self.schedule_embed = nn.Sequential(  # sched_dim comes from per‑program code
            nn.Linear(sched_dim, 128),
            nn.ReLU(),
        )
        self.final = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def embed_sparse(self, x, meta):
        y = self.stem(x)
        pools = []
        for stage in self.stages:
            y = stage(y)
            pools.append(self.glob_pool(y))
        pools = torch.cat(pools, dim=1)
        meta = self.meta(meta[:, :3])
        return self.matrix_embed(torch.cat([pools, meta], dim=1))

    def forward(self, x, meta, sched):
        m_emb = self.embed_sparse(x, meta)
        s_emb = self.schedule_embed(sched)
        return self.final(torch.cat([m_emb, s_emb], dim=1))
