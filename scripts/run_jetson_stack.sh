#!/usr/bin/env bash
# Convenience wrapper for the Jetson edge stack.
set -e
exec "$(cd "$(dirname "$0")/.." && pwd)/deployment/jetson/run_edge_stack.sh"
