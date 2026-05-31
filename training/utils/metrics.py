#!/usr/bin/env python3
"""Shared metric helpers used by training validation and offline evaluation."""
import numpy as np


def action_rmse(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def per_axis_mae(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    mae = np.mean(np.abs(pred - target), axis=0)
    return {'vx': float(mae[0]), 'vy': float(mae[1]), 'wz': float(mae[2])}


def command_smoothness(actions):
    """Mean L2 of consecutive command differences (lower is smoother)."""
    a = np.asarray(actions)
    if len(a) < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(a, axis=0), axis=1)))


def max_abs_error(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def mean_abs_error(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
