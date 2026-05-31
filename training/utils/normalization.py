#!/usr/bin/env python3
"""Input normalization helpers shared by training and the runtime.

The BEV channels already live in well-behaved ranges:
  - occupancy / mask channels in [0, 1]
  - confidence maps in [0, 1]
  - height channel in meters (roughly [z_min, z_max])
Goal vectors and velocities need scaling so the network sees ~unit inputs.

Keeping these constants in one place guarantees train-time and deploy-time use
identical normalization (a common source of sim-to-real gaps).
"""
import numpy as np

# goal = [dx, dy, distance, heading_error]
GOAL_SCALE = np.array([5.0, 5.0, 5.0, np.pi], dtype=np.float32)
# vel = [vx, vy, wz]
VEL_SCALE = np.array([0.6, 0.4, 1.0], dtype=np.float32)
# height channel index in the BEV tensor
HEIGHT_CHANNEL = 1
HEIGHT_RANGE = (-0.3, 1.2)


def normalize_goal(goal):
    return (np.asarray(goal, np.float32) / GOAL_SCALE).astype(np.float32)


def normalize_vel(vel):
    return (np.asarray(vel, np.float32) / VEL_SCALE).astype(np.float32)


def normalize_bev(bev):
    """Normalize only the height channel; others are already in [0,1]."""
    bev = np.asarray(bev, np.float32).copy()
    lo, hi = HEIGHT_RANGE
    if bev.shape[0] > HEIGHT_CHANNEL:
        bev[HEIGHT_CHANNEL] = np.clip(
            (bev[HEIGHT_CHANNEL] - lo) / (hi - lo), 0.0, 1.0)
    return bev


def denormalize_action(norm_action):
    """Network actions are already in velocity units (scaled by vmax)."""
    return np.asarray(norm_action, np.float32)
