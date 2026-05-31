#!/usr/bin/env python3
"""PyTorch Dataset for BEV behavior-cloning samples.

Supports two storage formats:
  1. .npz shards   : keys 'bev' (N,C,H,W), 'goal' (N,4), 'action' (N,3),
                     optional 'vel' (N,3). This is the recommended offline
                     format produced by inspect_dataset.py from rosbags.
  2. a directory   : loads and concatenates every *.npz inside it.

For environments without recorded data, ``SyntheticBevDataset`` generates
plausible samples from the same kinematic scene used by the RL env, so the BC
pipeline is runnable end-to-end out of the box (clearly labeled synthetic).
"""
import glob
import os

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _BASE = Dataset
except Exception:  # pragma: no cover
    _BASE = object

from .normalization import normalize_bev, normalize_goal, normalize_vel
from .augmentations import apply_random


class BevBCDataset(_BASE):
    def __init__(self, path, augment=True, use_velocity=True):
        self.augment = augment
        self.use_velocity = use_velocity
        self.rng = np.random.default_rng(0)
        files = ([path] if path.endswith('.npz')
                 else sorted(glob.glob(os.path.join(path, '*.npz'))))
        if not files:
            raise FileNotFoundError(f"No .npz dataset shards found at {path}")
        bevs, goals, acts, vels = [], [], [], []
        for f in files:
            d = np.load(f)
            bevs.append(d['bev'].astype(np.float32))
            goals.append(d['goal'].astype(np.float32))
            acts.append(d['action'].astype(np.float32))
            vels.append(d['vel'].astype(np.float32) if 'vel' in d
                        else np.zeros((len(d['action']), 3), np.float32))
        self.bev = np.concatenate(bevs, 0)
        self.goal = np.concatenate(goals, 0)
        self.action = np.concatenate(acts, 0)
        self.vel = np.concatenate(vels, 0)

    def __len__(self):
        return len(self.action)

    def __getitem__(self, i):
        bev = normalize_bev(self.bev[i])
        goal = self.goal[i]
        action = self.action[i]
        if self.augment:
            bev, goal, action = apply_random(bev, goal, action, self.rng)
        goal = normalize_goal(goal)
        vel = normalize_vel(self.vel[i])
        return {
            'bev': torch.from_numpy(np.ascontiguousarray(bev)),
            'goal': torch.from_numpy(goal),
            'vel': torch.from_numpy(vel),
            'action': torch.from_numpy(action),
        }


class SyntheticBevDataset(_BASE):
    """Generates synthetic BC samples via a simple expert controller.

    The 'expert' is a reactive go-to-goal-with-braking controller (the same
    spirit as DummyPolicy). This is ONLY for smoke-testing the training loop;
    it is not a substitute for real teleop/teacher data.
    """

    def __init__(self, n=2000, channels=13, size=100, use_velocity=True, seed=0):
        from ..models.rl_env_wrapper import KinematicBevEnv
        self.use_velocity = use_velocity
        env = KinematicBevEnv(bev_size=size, seed=seed)
        self.samples = []
        obs = env.reset()
        for _ in range(n):
            goal = obs['goal']
            # reactive expert: head to goal, brake if forward occupied
            occ = obs['bev'][0]
            fwd = occ[: size // 2].mean()
            vx = float(np.clip(0.6 * (1 - fwd * 3), 0, 0.6))
            vy = float(np.clip(0.5 * goal[1], -0.4, 0.4))
            wz = float(np.clip(1.5 * goal[3], -1.0, 1.0))
            action = np.array([vx, vy, wz], np.float32)
            self.samples.append((obs['bev'].copy(), goal.copy(),
                                 action, obs['vel'].copy()))
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        bev, goal, action, vel = self.samples[i]
        return {
            'bev': torch.from_numpy(np.ascontiguousarray(normalize_bev(bev))),
            'goal': torch.from_numpy(normalize_goal(goal)),
            'vel': torch.from_numpy(normalize_vel(vel)),
            'action': torch.from_numpy(action),
        }
