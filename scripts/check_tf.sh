#!/usr/bin/env bash
# Inspect the TF tree and key static transforms.
set -e
source /opt/ros/humble/setup.bash
WS="$(cd "$(dirname "$0")/../ros2_ws" && pwd)"
[ -f "${WS}/install/setup.bash" ] && source "${WS}/install/setup.bash"

echo "=== TF tree (writes frames.pdf if graphviz present) ==="
ros2 run tf2_tools view_frames || true
for child in lidar_link camera_link camera_depth_optical_frame; do
  echo "--- base_link -> $child ---"
  ros2 run tf2_ros tf2_echo base_link "$child" --once || echo "  (not available)"
done
