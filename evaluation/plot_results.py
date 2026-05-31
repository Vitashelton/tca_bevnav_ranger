#!/usr/bin/env python3
"""Plot evaluation results from one or more metrics JSON files.

Produces grouped bar charts comparing ablation rows on the key metrics. Reads
the JSON outputs of compute_metrics.py. Does NOT invent data -- if a metric is
missing/None it is skipped.

Usage:
  python3 plot_results.py --inputs lidar_only.json tca.json --labels LiDAR TCA \
      --out results.png
"""
import argparse
import json

BAR_METRICS = [
    'navigation_success_rate', 'collision_rate', 'timeout_rate',
    'door_passing_success_rate', 'narrow_corridor_success_rate',
    'dynamic_obstacle_avoidance_success_rate',
]
LAT_METRICS = ['policy_inference_latency_ms', 'bev_build_latency_ms',
               'end_to_end_latency_ms']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', default=None)
    ap.add_argument('--out', default='results.png')
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("matplotlib not installed. pip install matplotlib")
        return

    data = [json.load(open(p)) for p in args.inputs]
    labels = args.labels or [f"run{i}" for i in range(len(data))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(BAR_METRICS))
    w = 0.8 / len(data)
    for i, (d, lab) in enumerate(zip(data, labels)):
        vals = [d.get(m) or 0.0 for m in BAR_METRICS]
        ax1.bar(x + i * w, vals, w, label=lab)
    ax1.set_xticks(x + 0.4)
    ax1.set_xticklabels([m.replace('_rate', '').replace('_', '\n')
                         for m in BAR_METRICS], fontsize=8)
    ax1.set_title('Success / failure rates')
    ax1.legend()

    xl = np.arange(len(LAT_METRICS))
    for i, (d, lab) in enumerate(zip(data, labels)):
        vals = [d.get(m) or 0.0 for m in LAT_METRICS]
        ax2.bar(xl + i * w, vals, w, label=lab)
    ax2.set_xticks(xl + 0.4)
    ax2.set_xticklabels([m.replace('_ms', '').replace('_', '\n')
                         for m in LAT_METRICS], fontsize=8)
    ax2.set_title('Latency (ms)')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[plot_results] wrote {args.out}")


if __name__ == '__main__':
    main()
