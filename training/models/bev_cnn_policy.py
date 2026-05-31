#!/usr/bin/env python3
"""Lightweight BEV CNN policy (<5M params target).

Inputs:
  bev   : (B, C, H, W)  TCA-BEV tensor
  goal  : (B, 4)        [dx, dy, distance, heading_error]
  vel   : (B, 3)        optional current velocity (vx, vy, wz)
Output:
  (B, 4)  -> vx, vy, wz, uncertainty
  vx,vy,wz are tanh-squashed and scaled by max velocities.

The model is intentionally small so it exports cleanly to ONNX/TensorRT
and runs in real time on a Jetson Orin Nano.
"""
import torch
import torch.nn as nn


class BevEncoder(nn.Module):
    def __init__(self, in_ch, feat=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, feat)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return torch.relu(self.fc(x))


class GoalEncoder(nn.Module):
    def __init__(self, in_dim=4, feat=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, feat), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class BevCnnPolicy(nn.Module):
    def __init__(self, in_channels=13, use_velocity=True,
                 max_vx=0.6, max_vy=0.4, max_wz=1.0):
        super().__init__()
        self.use_velocity = use_velocity
        self.register_buffer('vmax', torch.tensor([max_vx, max_vy, max_wz]))
        self.bev_enc = BevEncoder(in_channels, feat=128)
        self.goal_enc = GoalEncoder(4, feat=32)
        extra = 3 if use_velocity else 0
        fuse_dim = 128 + 32 + extra
        self.trunk = nn.Sequential(
            nn.Linear(fuse_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
        )
        self.action_head = nn.Linear(64, 3)       # vx, vy, wz (pre-tanh)
        self.uncertainty_head = nn.Linear(64, 1)  # >=0 via softplus

    def forward(self, bev, goal, vel=None):
        bf = self.bev_enc(bev)
        gf = self.goal_enc(goal)
        if self.use_velocity:
            if vel is None:
                vel = torch.zeros(bev.shape[0], 3, device=bev.device)
            feat = torch.cat([bf, gf, vel], dim=1)
        else:
            feat = torch.cat([bf, gf], dim=1)
        h = self.trunk(feat)
        act = torch.tanh(self.action_head(h)) * self.vmax
        unc = torch.nn.functional.softplus(self.uncertainty_head(h))
        return torch.cat([act, unc], dim=1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    m = BevCnnPolicy(in_channels=13)
    print(f"params = {count_params(m)/1e6:.3f}M")
    b = torch.randn(2, 13, 100, 100)
    g = torch.randn(2, 4)
    v = torch.randn(2, 3)
    print(m(b, g, v).shape)
