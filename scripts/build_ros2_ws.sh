#!/usr/bin/env bash
# Build the ROS2 workspace.
set -e
WS="$(cd "$(dirname "$0")/../ros2_ws" && pwd)"
cd "${WS}"
source /opt/ros/humble/setup.bash
colcon build --symlink-install "$@"
echo "Build done. Now: source ${WS}/install/setup.bash"
