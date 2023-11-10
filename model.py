import torch.nn as nn
import torch


class Reward(nn.Module):
    def __init__(self, view_dim, feat_dim):
        super().__init__()
        self.view_blk = nn.Sequential(
            nn.Conv2d(view_dim, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 2, 1, 0),
            nn.ReLU(),
            nn.Flatten(start_dim=1)
        )
        self.feat_blk = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, view, feat):
        view_emb = self.view_blk(view)
        feat_emb = self.feat_blk(feat)
        view_feat = torch.cat([view_emb, feat_emb], axis=1)
        logits = self.fc(view_feat)
        return logits
