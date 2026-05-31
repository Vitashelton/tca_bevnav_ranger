#!/usr/bin/env python3
"""Export a trained BevCnnPolicy checkpoint to ONNX.

Usage:
  python -m training.scripts.export_onnx --ckpt training/runs/bc/policy_bc.pt \
      --out deployment/models/policy.onnx

The exported graph has fixed input shapes (batch=1) so it converts cleanly to a
TensorRT engine on the Jetson Orin Nano. Inputs: bev, goal, vel. Output: a
single (1,4) tensor [vx, vy, wz, uncertainty].
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from training.models.bev_cnn_policy import BevCnnPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='deployment/models/policy.onnx')
    ap.add_argument('--channels', type=int, default=13)
    ap.add_argument('--size', type=int, default=100)
    ap.add_argument('--opset', type=int, default=17)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg = ckpt.get('config', {})
    model = BevCnnPolicy(
        in_channels=cfg.get('in_channels', args.channels),
        use_velocity=cfg.get('use_velocity', True),
        max_vx=cfg.get('max_vx', 0.6), max_vy=cfg.get('max_vy', 0.4),
        max_wz=cfg.get('max_wz', 1.0))
    model.load_state_dict(ckpt.get('model', ckpt))
    model.eval()

    bev = torch.randn(1, args.channels, args.size, args.size)
    goal = torch.randn(1, 4)
    vel = torch.randn(1, 3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.onnx.export(
        model, (bev, goal, vel), args.out,
        input_names=['bev', 'goal', 'vel'], output_names=['action'],
        opset_version=args.opset, dynamic_axes=None)
    print(f"[export_onnx] wrote {args.out}")


if __name__ == '__main__':
    main()
