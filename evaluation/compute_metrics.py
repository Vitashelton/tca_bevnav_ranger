#!/usr/bin/env python3
"""Compute navigation metrics from per-episode CSV logs.

Input CSV (one row per control step) is expected to contain at least:
  episode, t, x, y, vx, vy, wz, min_obstacle_distance, event,
  policy_latency_ms, bev_latency_ms, e2e_latency_ms
where `event` is one of: step|success|collision|timeout and may also include
door_passed / corridor scenario tags.

This script ONLY computes metrics from logged data. It does not fabricate or
assume any success/collision numbers. Run it on real or simulated logs.

Usage:
  python3 compute_metrics.py --csv runs/exp1.csv --out runs/exp1_metrics.json
"""
import argparse
import csv
import json
import math
from collections import defaultdict


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def group_episodes(rows):
    eps = defaultdict(list)
    for r in rows:
        eps[r.get('episode', '0')].append(r)
    return eps


def episode_outcome(ep_rows):
    events = [r.get('event', 'step') for r in ep_rows]
    for outcome in ('collision', 'success', 'timeout'):
        if outcome in events:
            return outcome
    return 'timeout'


def path_length(ep_rows):
    d = 0.0
    for a, b in zip(ep_rows, ep_rows[1:]):
        d += math.hypot(_f(b, 'x') - _f(a, 'x'), _f(b, 'y') - _f(a, 'y'))
    return d


def travel_time(ep_rows):
    if len(ep_rows) < 2:
        return 0.0
    return _f(ep_rows[-1], 't') - _f(ep_rows[0], 't')


def compute(rows):
    eps = group_episodes(rows)
    n = len(eps)
    outcomes = {k: episode_outcome(v) for k, v in eps.items()}
    success = sum(o == 'success' for o in outcomes.values())
    collision = sum(o == 'collision' for o in outcomes.values())
    timeout = sum(o == 'timeout' for o in outcomes.values())

    speeds, smooth, min_dists, times, lengths = [], [], [], [], []
    pol_lat, bev_lat, e2e_lat = [], [], []
    for ep in eps.values():
        times.append(travel_time(ep))
        lengths.append(path_length(ep))
        for a, b in zip(ep, ep[1:]):
            dv = math.sqrt((_f(b, 'vx') - _f(a, 'vx')) ** 2
                           + (_f(b, 'vy') - _f(a, 'vy')) ** 2
                           + (_f(b, 'wz') - _f(a, 'wz')) ** 2)
            smooth.append(dv)
        for r in ep:
            speeds.append(math.hypot(_f(r, 'vx'), _f(r, 'vy')))
            if 'min_obstacle_distance' in r:
                min_dists.append(_f(r, 'min_obstacle_distance'))
            pol_lat.append(_f(r, 'policy_latency_ms'))
            bev_lat.append(_f(r, 'bev_latency_ms'))
            e2e_lat.append(_f(r, 'e2e_latency_ms'))

    def mean(x):
        return sum(x) / len(x) if x else 0.0

    # scenario-specific success (rows may tag scenario)
    def scenario_success(tag):
        sel = {k: v for k, v in eps.items()
               if any(tag in (r.get('scenario', '')) for r in v)}
        if not sel:
            return None
        s = sum(episode_outcome(v) == 'success' for v in sel.values())
        return s / len(sel)

    return {
        'episodes': n,
        'navigation_success_rate': success / n if n else 0.0,
        'collision_rate': collision / n if n else 0.0,
        'timeout_rate': timeout / n if n else 0.0,
        'average_travel_time': mean(times),
        'path_length': mean(lengths),
        'minimum_obstacle_distance': min(min_dists) if min_dists else 0.0,
        'average_speed': mean(speeds),
        'command_smoothness': mean(smooth),
        'policy_inference_latency_ms': mean(pol_lat),
        'bev_build_latency_ms': mean(bev_lat),
        'end_to_end_latency_ms': mean(e2e_lat),
        'door_passing_success_rate': scenario_success('door'),
        'narrow_corridor_success_rate': scenario_success('corridor'),
        'dynamic_obstacle_avoidance_success_rate': scenario_success('dynamic'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    rows = load_rows(args.csv)
    metrics = compute(rows)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[compute_metrics] wrote {args.out}")


if __name__ == '__main__':
    main()
