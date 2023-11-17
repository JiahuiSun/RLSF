import torch.nn as nn
import torch

# TODO: 用ResNet处理现在的图片大小
class RewardModel(nn.Module):
    """
    input shape:
        state: (B, T, 3, 192, 256)
        action: (B, T, 4)
    output shape:
        reward: (B, T)
    """
    def __init__(self, state_dim=3, act_dim=4):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Conv2d(state_dim, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 2, 1, 0),
            nn.ReLU(),
            nn.Flatten(start_dim=1)
        )
        self.act_encoder = nn.Sequential(
            nn.Linear(act_dim, 128),
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
        state_emb = self.state_encoder(view)
        act_emb = self.act_encoder(feat)
        state_act = torch.cat([state_emb, act_emb], axis=1)
        logits = self.fc(state_act)
        return logits
