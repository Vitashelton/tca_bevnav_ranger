#!/usr/bin/env python3
"""PPO fine-tuning of the TCA-BEV policy on the lightweight kinematic env.

This is a compact, self-contained PPO (single-file, clip objective, GAE). It
is intended to *fine-tune* a BC-initialized policy, matching the project's
teacher-student strategy (BC warm-start -> PPO refinement).

Usage:
  python -m training.scripts.train_ppo --config training/configs/ppo.yaml \
      --init training/runs/bc/policy_bc.pt

Reward terms are defined in training/models/rl_env_wrapper.py (RewardConfig)
and documented in docs/training_strategy.md.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    import yaml
except Exception:
    yaml = None
import torch
import torch.nn as nn

from training.models.bev_cnn_policy import BevEncoder, GoalEncoder
from training.models.policy_heads import GaussianActionHead
from training.models.rl_env_wrapper import KinematicBevEnv
from training.utils.normalization import normalize_bev, normalize_goal, normalize_vel
from training.utils.logger import TrainLogger


class ActorCritic(nn.Module):
    """Shares the BEV/goal encoders with the BC model; adds a value head."""

    def __init__(self, in_channels=13, max_v=(0.6, 0.4, 1.0)):
        super().__init__()
        self.bev_enc = BevEncoder(in_channels, feat=128)
        self.goal_enc = GoalEncoder(4, feat=32)
        self.trunk = nn.Sequential(
            nn.Linear(128 + 32 + 3, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU())
        self.actor = GaussianActionHead(64, *max_v)
        self.critic = nn.Linear(64, 1)

    def _feat(self, bev, goal, vel):
        f = torch.cat([self.bev_enc(bev), self.goal_enc(goal), vel], dim=1)
        return self.trunk(f)

    def act(self, bev, goal, vel):
        h = self._feat(bev, goal, vel)
        action, logp = self.actor.sample(h)
        value = self.critic(h).squeeze(-1)
        return action, logp, value

    def evaluate(self, bev, goal, vel, action):
        h = self._feat(bev, goal, vel)
        logp = self.actor.log_prob(h, action)
        value = self.critic(h).squeeze(-1)
        return logp, value


def load_cfg(path):
    d = {'in_channels': 13, 'total_steps': 50000, 'rollout': 1024,
         'epochs': 4, 'minibatch': 256, 'gamma': 0.99, 'lam': 0.95,
         'clip': 0.2, 'lr': 3e-4, 'ent_coef': 0.0, 'vf_coef': 0.5,
         'logdir': 'training/runs/ppo', 'ckpt': 'training/runs/ppo/policy_ppo.pt'}
    if path and yaml and os.path.exists(path):
        with open(path) as f:
            d.update(yaml.safe_load(f) or {})
    return d


def obs_to_tensors(obs, device):
    bev = torch.from_numpy(np.ascontiguousarray(normalize_bev(obs['bev'])))[None].to(device)
    goal = torch.from_numpy(normalize_goal(obs['goal']))[None].to(device)
    vel = torch.from_numpy(normalize_vel(obs['vel']))[None].to(device)
    return bev, goal, vel


def compute_gae(rewards, values, dones, gamma, lam, last_value):
    adv = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nextv = last_value if t == len(rewards) - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nextv * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    return adv, adv + values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='training/configs/ppo.yaml')
    ap.add_argument('--init', default=None, help='BC checkpoint to warm-start')
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[train_ppo] device={device} cfg={cfg}")

    env = KinematicBevEnv(seed=0)
    model = ActorCritic(in_channels=cfg['in_channels']).to(device)
    if args.init and os.path.exists(args.init):
        ckpt = torch.load(args.init, map_location=device)
        sd = ckpt.get('model', ckpt)
        # load only matching encoder/trunk weights
        own = model.state_dict()
        loaded = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
        own.update(loaded)
        model.load_state_dict(own)
        print(f"[train_ppo] warm-started {len(loaded)} tensors from {args.init}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    logger = TrainLogger(cfg['logdir'])

    obs = env.reset()
    global_step = 0
    ep_ret, ep_rets = 0.0, []
    while global_step < cfg['total_steps']:
        buf = {k: [] for k in ('bev', 'goal', 'vel', 'act', 'logp', 'val', 'rew', 'done')}
        for _ in range(cfg['rollout']):
            bev, goal, vel = obs_to_tensors(obs, device)
            with torch.no_grad():
                action, logp, value = model.act(bev, goal, vel)
            a = action.cpu().numpy()[0]
            nobs, rew, done, info = env.step(a)
            buf['bev'].append(normalize_bev(obs['bev']))
            buf['goal'].append(normalize_goal(obs['goal']))
            buf['vel'].append(normalize_vel(obs['vel']))
            buf['act'].append(a)
            buf['logp'].append(float(logp))
            buf['val'].append(float(value))
            buf['rew'].append(rew)
            buf['done'].append(float(done))
            ep_ret += rew
            obs = nobs
            global_step += 1
            if done:
                ep_rets.append(ep_ret)
                ep_ret = 0.0
                obs = env.reset()

        with torch.no_grad():
            bev, goal, vel = obs_to_tensors(obs, device)
            _, _, last_v = model.act(bev, goal, vel)
        adv, ret = compute_gae(np.array(buf['rew']), np.array(buf['val']),
                               np.array(buf['done']), cfg['gamma'], cfg['lam'],
                               float(last_v))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        bev_t = torch.from_numpy(np.ascontiguousarray(np.stack(buf['bev']))).to(device)
        goal_t = torch.from_numpy(np.stack(buf['goal'])).to(device)
        vel_t = torch.from_numpy(np.stack(buf['vel'])).to(device)
        act_t = torch.from_numpy(np.stack(buf['act']).astype(np.float32)).to(device)
        old_logp = torch.tensor(buf['logp'], device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)

        n = len(buf['rew'])
        for _ in range(cfg['epochs']):
            idx = np.random.permutation(n)
            for s in range(0, n, cfg['minibatch']):
                mb = idx[s:s + cfg['minibatch']]
                logp, value = model.evaluate(bev_t[mb], goal_t[mb], vel_t[mb], act_t[mb])
                ratio = torch.exp(logp - old_logp[mb])
                clip = torch.clamp(ratio, 1 - cfg['clip'], 1 + cfg['clip'])
                pg = -torch.min(ratio * adv_t[mb], clip * adv_t[mb]).mean()
                vf = (value - ret_t[mb]).pow(2).mean()
                loss = pg + cfg['vf_coef'] * vf
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
        mean_ret = float(np.mean(ep_rets[-10:])) if ep_rets else 0.0
        logger.log(global_step, {'mean_ep_return': mean_ret,
                                 'pg_loss': float(pg), 'vf_loss': float(vf)})
        print(f"[train_ppo] step {global_step} mean_return={mean_ret:.2f}")

    os.makedirs(os.path.dirname(cfg['ckpt']), exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': cfg}, cfg['ckpt'])
    print(f"[train_ppo] saved -> {cfg['ckpt']}")
    logger.close()


if __name__ == '__main__':
    main()
