#!/usr/bin/env bash
# Quick health check: list key topics and their publish rates.
set -e
source /opt/ros/humble/setup.bash
WS="$(cd "$(dirname "$0")/../ros2_ws" && pwd)"
[ -f "${WS}/install/setup.bash" ] && source "${WS}/install/setup.bash"

echo "=== ros2 topic list ==="
ros2 topic list
for t in /sensors/lidar_points /sensors/depth_image /anchor/odom \
         /bev/tensor /goal/vector /cmd_vel_raw /cmd_vel_safe /safety/status; do
  echo "--- hz: $t (5s) ---"
  timeout 5 ros2 topic hz "$t" || echo "  (no messages)"
done
