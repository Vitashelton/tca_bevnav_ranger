#!/usr/bin/env python3
"""Inspect a dataset and/or extract rosbags into .npz BC shards.

Two modes:
  --bag <dir>  : read a rosbag2 directory, extract (bev, goal, action)
                 nearest-time-matched samples, write a .npz shard.
  --npz <path> : print shape/stats of an existing .npz shard or directory.

Extraction requires a sourced ROS2 environment (see rosbag_loader). The .npz
shards are the portable format consumed by training on machines without ROS.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def extract(bag, out):
    from training.utils.rosbag_loader import load_bag
    samples = load_bag(bag)
    if not samples:
        print("No matched samples found in bag.")
        return
    bev = np.stack([s['bev'] for s in samples]).astype(np.float32)
    goal = np.stack([s['goal'] for s in samples]).astype(np.float32)
    action = np.stack([s['action'] for s in samples]).astype(np.float32)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    np.savez_compressed(out, bev=bev, goal=goal, action=action)
    print(f"[inspect_dataset] wrote {len(samples)} samples -> {out}")


def stats(path):
    import glob
    files = [path] if path.endswith('.npz') else sorted(glob.glob(os.path.join(path, '*.npz')))
    total = 0
    for f in files:
        d = np.load(f)
        n = len(d['action'])
        total += n
        print(f"{f}: bev{d['bev'].shape} goal{d['goal'].shape} "
              f"action{d['action'].shape}")
        print(f"    action mean={d['action'].mean(0)} std={d['action'].std(0)}")
    print(f"[inspect_dataset] total samples = {total}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', default=None)
    ap.add_argument('--npz', default=None)
    ap.add_argument('--out', default='training/datasets/extracted/shard_000.npz')
    args = ap.parse_args()
    if args.bag:
        extract(args.bag, args.out)
    elif args.npz:
        stats(args.npz)
    else:
        print("Provide --bag <rosbag_dir> or --npz <path>")


if __name__ == '__main__':
    main()
