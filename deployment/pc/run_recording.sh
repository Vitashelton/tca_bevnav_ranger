#!/usr/bin/env bash
# Record a dataset session from the PC.
set -e
WS="$(cd "$(dirname "$0")/../../ros2_ws" && pwd)"
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
SCENARIO="${SCENARIO:-session}"
exec ros2 launch tca_bev_bringup record_dataset.launch.py scenario:="${SCENARIO}"
