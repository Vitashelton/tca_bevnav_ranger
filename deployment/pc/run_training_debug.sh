#!/usr/bin/env bash
# Convenience: run a short synthetic BC training to verify the pipeline on PC.
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
python3 -m training.scripts.train_bc --synthetic --epochs 2
