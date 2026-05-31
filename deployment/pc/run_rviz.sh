#!/usr/bin/env bash
# Visualize the running stack from the PC (RViz + BEV markers).
set -e
WS="$(cd "$(dirname "$0")/../../ros2_ws" && pwd)"
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
exec ros2 launch ranger_bringup pc_visualization.launch.py
