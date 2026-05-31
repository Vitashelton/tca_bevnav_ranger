#!/usr/bin/env python3
"""Minimal training logger: CSV always, TensorBoard if available.

Avoids a hard dependency on tensorboard so the code runs in bare environments.
"""
import csv
import os
import time

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:  # pragma: no cover - tensorboard optional
    _HAS_TB = False


class TrainLogger:
    def __init__(self, logdir, use_tb=True):
        os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir
        self.csv_path = os.path.join(logdir, 'metrics.csv')
        self._csv_file = open(self.csv_path, 'w', newline='')
        self._writer = None
        self.tb = SummaryWriter(logdir) if (use_tb and _HAS_TB) else None
        self._t0 = time.time()

    def log(self, step, metrics: dict):
        row = {'step': step, 'wall_time': round(time.time() - self._t0, 2)}
        row.update({k: float(v) for k, v in metrics.items()})
        if self._writer is None:
            self._writer = csv.DictWriter(self._csv_file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._csv_file.flush()
        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, float(v), step)

    def close(self):
        self._csv_file.close()
        if self.tb:
            self.tb.close()
