#!/usr/bin/env python3
"""Compare PyTorch vs ONNX outputs for the exported policy.

Usage:
  python -m training.scripts.validate_policy --ckpt ... --onnx ...

Reports max/mean absolute error over random inputs. Requires onnxruntime; if
it is missing the script explains how to install it instead of failing
silently.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from training.models.bev_cnn_policy import BevCnnPolicy
from training.utils.metrics import max_abs_error, mean_abs_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--channels', type=int, default=13)
    ap.add_argument('--size', type=int, default=100)
    ap.add_argument('--trials', type=int, default=20)
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except Exception:
        print("onnxruntime not installed. pip install onnxruntime")
        return

    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg = ckpt.get('config', {})
    model = BevCnnPolicy(in_channels=cfg.get('in_channels', args.channels),
                         use_velocity=cfg.get('use_velocity', True))
    model.load_state_dict(ckpt.get('model', ckpt))
    model.eval()
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])

    maxe, meane = [], []
    for _ in range(args.trials):
        bev = np.random.randn(1, args.channels, args.size, args.size).astype(np.float32)
        goal = np.random.randn(1, 4).astype(np.float32)
        vel = np.random.randn(1, 3).astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(bev), torch.from_numpy(goal),
                        torch.from_numpy(vel)).numpy()
        out = sess.run(['action'], {'bev': bev, 'goal': goal, 'vel': vel})[0]
        maxe.append(max_abs_error(ref, out))
        meane.append(mean_abs_error(ref, out))
    print(f"[validate_policy] over {args.trials} trials: "
          f"max_err={max(maxe):.3e}  mean_err={np.mean(meane):.3e}")
    if max(maxe) > 1e-3:
        print("WARNING: error > 1e-3; check opset / unsupported ops.")


if __name__ == '__main__':
    main()
