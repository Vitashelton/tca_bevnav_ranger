#!/usr/bin/env bash
# Convenience wrapper for PC RViz visualization.
set -e
exec "$(cd "$(dirname "$0")/.." && pwd)/deployment/pc/run_rviz.sh"
