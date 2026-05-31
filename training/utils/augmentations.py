#!/usr/bin/env python3
"""Data augmentations for BEV behavior-cloning samples.

All augmentations operate on a (bev, goal, action) triple and keep them
consistent. The most important one is the lateral mirror: because the base is
omni-directional, mirroring the y axis also flips vy, wz and the goal's dy /
heading_error, doubling the effective data without changing semantics.
"""
import numpy as np


def mirror_y(bev, goal, action):
    """Flip the BEV across its vertical center column and the y-components."""
    bev_m = bev[:, :, ::-1].copy()
    # goal = [dx, dy, distance, heading_error]
    goal_m = goal.copy()
    goal_m[1] = -goal_m[1]
    goal_m[3] = -goal_m[3]
    # action = [vx, vy, wz]
    act_m = action.copy()
    act_m[1] = -act_m[1]
    act_m[2] = -act_m[2]
    # goal_direction_map channels (11=x stays, 12=y flips sign)
    if bev_m.shape[0] > 12:
        bev_m[12] = -bev_m[12]
    return bev_m, goal_m, act_m


def add_bev_dropout(bev, p=0.02, rng=None):
    """Randomly zero a fraction of occupancy cells to mimic sparse returns."""
    rng = rng or np.random.default_rng()
    mask = rng.random(bev.shape[1:]) < p
    out = bev.copy()
    out[0][mask] = 0.0
    return out


def jitter_confidence(bev, sigma=0.03, rng=None):
    """Add small noise to the confidence channels (8,9,10) -> robustness."""
    rng = rng or np.random.default_rng()
    out = bev.copy()
    for ch in (8, 9, 10):
        if out.shape[0] > ch:
            out[ch] = np.clip(out[ch] + rng.normal(0, sigma, out[ch].shape), 0, 1)
    return out


def apply_random(bev, goal, action, rng=None):
    rng = rng or np.random.default_rng()
    if rng.random() < 0.5:
        bev, goal, action = mirror_y(bev, goal, action)
    if rng.random() < 0.5:
        bev = add_bev_dropout(bev, rng=rng)
    if rng.random() < 0.5:
        bev = jitter_confidence(bev, rng=rng)
    return bev, goal, action
