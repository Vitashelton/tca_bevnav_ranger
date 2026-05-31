#!/usr/bin/env python3
"""Reusable output heads for the BEV policy network.

These are split out from ``bev_cnn_policy`` so that BC and PPO can share the
same action parameterization. Keeping the heads tiny is important: the whole
network must export to ONNX and run on a Jetson Orin Nano in real time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicActionHead(nn.Module):
    """Outputs a squashed, velocity-scaled action used by behavior cloning.

    Action layout: [vx, vy, wz]. Each is tanh-squashed then scaled by the
    per-axis maximum velocity, guaranteeing the network can never request a
    command outside the mechanical limits of the Ranger Mini 2.0.
    """

    def __init__(self, in_dim, max_vx=0.6, max_vy=0.4, max_wz=1.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, 3)
        self.register_buffer('vmax', torch.tensor([max_vx, max_vy, max_wz]))

    def forward(self, x):
        return torch.tanh(self.fc(x)) * self.vmax


class UncertaintyHead(nn.Module):
    """Predicts a non-negative scalar uncertainty (softplus).

    The runtime feeds this to the safety supervisor as an extra speed-scaling
    signal: high predicted uncertainty -> lower allowed speed.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return F.softplus(self.fc(x))


class GaussianActionHead(nn.Module):
    """Stochastic head for PPO.

    Produces a tanh-squashed Gaussian over [vx, vy, wz]. The mean is scaled by
    the velocity limits; log_std is a learned, state-independent parameter
    which is the common, stable choice for continuous-control PPO.
    """

    def __init__(self, in_dim, max_vx=0.6, max_vy=0.4, max_wz=1.0,
                 log_std_init=-0.5):
        super().__init__()
        self.mean_fc = nn.Linear(in_dim, 3)
        self.log_std = nn.Parameter(torch.ones(3) * log_std_init)
        self.register_buffer('vmax', torch.tensor([max_vx, max_vy, max_wz]))

    def forward(self, x):
        mean = self.mean_fc(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def sample(self, x):
        """Returns (action, log_prob). action is in velocity units."""
        mean, std = self.forward(x)
        normal = torch.distributions.Normal(mean, std)
        raw = normal.rsample()
        log_prob = normal.log_prob(raw).sum(-1)
        # tanh squash correction
        squashed = torch.tanh(raw)
        log_prob -= torch.log(1 - squashed.pow(2) + 1e-6).sum(-1)
        return squashed * self.vmax, log_prob

    def log_prob(self, x, action):
        """Log-prob of a velocity-unit action under the current policy."""
        mean, std = self.forward(x)
        squashed = (action / self.vmax).clamp(-0.999, 0.999)
        raw = torch.atanh(squashed)
        normal = torch.distributions.Normal(mean, std)
        lp = normal.log_prob(raw).sum(-1)
        lp -= torch.log(1 - squashed.pow(2) + 1e-6).sum(-1)
        return lp
