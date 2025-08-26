import torch.nn as nn


class EnvEncoder2D(nn.Module):
    def __init__(self, in_ch=1, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, emb_dim)

    def forward(self, env_grid):  # [B,1,H,W]
        h = self.net(env_grid).flatten(1)  # [B,64]
        return self.fc(h)


class EnvEncoder3D(nn.Module):
    def __init__(self, in_ch=1, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(32, emb_dim)

    def forward(self, env_grid):
        h = self.net(env_grid).flatten(1)
        return self.fc(h)


class ScaleEmbedding(nn.Module):
    def __init__(self, num_scales: int, emb_dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(num_scales, emb_dim)

    def forward(self, scale_id):  # scale_id: [B]
        return self.emb(scale_id)  # [B, emb_dim]
