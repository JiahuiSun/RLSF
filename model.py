import torch.nn as nn
import torch
from torch.nn import functional as F


class AlexNet(nn.Module):
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


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class ResNet18(nn.Module):
    """ResNet18，https://zh-v2.d2l.ai/chapter_convolutional-modern/resnet.html
    ResNet18的维度变化情况：
    (3, 192, 256) -> (64, 96, 128) -> (64, 48, 64) -> (64, 48, 64) -> (128, 24, 32) -> (256, 12, 16) -> (512, 6, 8) -> 512 -> 128

    input shape:
        state: (B, T, 3, 192, 256)，先把输入变成(BT, ...)，得到结果再reshape成(B, T)
        action: (B, T, 4)
    output shape:
        reward: (B, T)
    """
    def __init__(self, state_dim=3, act_dim=4):
        super().__init__()
        b1 = nn.Sequential(
            nn.Conv2d(state_dim, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.state_encoder = nn.Sequential(
            b1, b2, b3, b4, b5,
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 128)
        )
        self.act_encoder = nn.Sequential(
            nn.Linear(act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, act):
        B, T, C, H, W = obs.size()
        B, T, A = act.size()
        obs = obs.view(B*T, C, H, W)
        act = act.view(B*T, A)
        state_emb = self.state_encoder(obs)
        act_emb = self.act_encoder(act)
        state_act = torch.cat([state_emb, act_emb], axis=1)
        logits = self.fc(state_act)
        logits = logits.view(B, T)
        return logits
