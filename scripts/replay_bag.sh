#!/usr/bin/env bash
# Replay a recorded bag (optionally remapping clock).
set -e
source /opt/ros/humble/setup.bash
WS="$(cd "$(dirname "$0")/../ros2_ws" && pwd)"
[ -f "${WS}/install/setup.bash" ] && source "${WS}/install/setup.bash"
BAG="${1:?usage: replay_bag.sh <bag_dir> [rate]}"
RATE="${2:-1.0}"
ros2 bag play "${BAG}" --rate "${RATE}"
