#!/usr/bin/env python3
"""Behavior-cloning trainer for the TCA-BEV policy.

Usage:
  python -m training.scripts.train_bc --config training/configs/bc.yaml
  # or, from the training/ dir with PYTHONPATH set, run directly.

If no dataset is provided (or --synthetic is passed) a synthetic dataset is
used so the loop is runnable without recorded data. Real runs should point
--data at a directory of .npz shards extracted from rosbags.
"""
import argparse
import os
import sys

import numpy as np

# allow running both as module and as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    import yaml
except Exception:
    yaml = None
import torch
from torch.utils.data import DataLoader

from training.models.bev_cnn_policy import BevCnnPolicy, count_params
from training.models.losses import bc_total_loss
from training.utils.bev_dataset import BevBCDataset, SyntheticBevDataset
from training.utils.logger import TrainLogger
from training.utils.metrics import action_rmse, per_axis_mae


def load_config(path):
    default = {
        'in_channels': 13, 'use_velocity': True,
        'max_vx': 0.6, 'max_vy': 0.4, 'max_wz': 1.0,
        'epochs': 20, 'batch_size': 64, 'lr': 3e-4, 'val_split': 0.1,
        'weights': {'action': 1.0, 'smooth': 0.05,
                    'collision': 0.5, 'uncertainty': 0.1},
        'logdir': 'training/runs/bc', 'ckpt': 'training/runs/bc/policy_bc.pt',
    }
    if path and yaml and os.path.exists(path):
        with open(path) as f:
            default.update(yaml.safe_load(f) or {})
    return default


def split_dataset(ds, val_split):
    n = len(ds)
    n_val = max(1, int(n * val_split))
    idx = np.random.default_rng(0).permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (torch.utils.data.Subset(ds, train_idx),
            torch.utils.data.Subset(ds, val_idx))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, tgts = [], []
    for batch in loader:
        bev = batch['bev'].to(device)
        goal = batch['goal'].to(device)
        vel = batch['vel'].to(device)
        out = model(bev, goal, vel)
        preds.append(out[:, :3].cpu().numpy())
        tgts.append(batch['action'].numpy())
    preds, tgts = np.concatenate(preds), np.concatenate(tgts)
    return action_rmse(preds, tgts), per_axis_mae(preds, tgts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='training/configs/bc.yaml')
    ap.add_argument('--data', default=None, help='dir or .npz of BC samples')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--epochs', type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg['epochs'] = args.epochs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[train_bc] device={device}  config={cfg}")

    if args.data and not args.synthetic:
        ds = BevBCDataset(args.data, augment=True,
                          use_velocity=cfg['use_velocity'])
    else:
        print("[train_bc] WARNING: using SYNTHETIC dataset (smoke test only).")
        ds = SyntheticBevDataset(n=2000, channels=cfg['in_channels'],
                                 use_velocity=cfg['use_velocity'])
    train_ds, val_ds = split_dataset(ds, cfg['val_split'])
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'])

    model = BevCnnPolicy(in_channels=cfg['in_channels'],
                         use_velocity=cfg['use_velocity'],
                         max_vx=cfg['max_vx'], max_vy=cfg['max_vy'],
                         max_wz=cfg['max_wz']).to(device)
    print(f"[train_bc] params = {count_params(model)/1e6:.3f}M")
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    logger = TrainLogger(cfg['logdir'])

    step = 0
    for epoch in range(cfg['epochs']):
        model.train()
        for batch in train_loader:
            bev = batch['bev'].to(device)
            goal = batch['goal'].to(device)
            vel = batch['vel'].to(device)
            target = batch['action'].to(device)
            out = model(bev, goal, vel)
            loss, parts = bc_total_loss(out, target, bev, cfg['weights'])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 20 == 0:
                logger.log(step, parts)
            step += 1
        rmse, mae = evaluate(model, val_loader, device)
        logger.log(step, {'val_rmse': rmse, 'val_mae_vx': mae['vx'],
                          'val_mae_vy': mae['vy'], 'val_mae_wz': mae['wz']})
        print(f"[train_bc] epoch {epoch+1}/{cfg['epochs']} "
              f"val_rmse={rmse:.4f} mae={mae}")

    os.makedirs(os.path.dirname(cfg['ckpt']), exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': cfg}, cfg['ckpt'])
    print(f"[train_bc] saved checkpoint -> {cfg['ckpt']}")
    logger.close()


if __name__ == '__main__':
    main()
