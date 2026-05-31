#!/usr/bin/env bash
# Record all topics needed for training and evaluation.
set -e
source /opt/ros/humble/setup.bash
WS="$(cd "$(dirname "$0")/../ros2_ws" && pwd)"
[ -f "${WS}/install/setup.bash" ] && source "${WS}/install/setup.bash"

SCENARIO="${1:-session}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="datasets/${SCENARIO}_${STAMP}"
mkdir -p "$(dirname "$OUT")"

ros2 bag record -o "${OUT}" \
  /sensors/lidar_points /sensors/depth_image /sensors/color_image \
  /sensors/camera_info /aligned/lidar_points /aligned/depth_image \
  /anchor/odom /anchor/status /bev/tensor /bev/image_debug \
  /goal/vector /cmd_vel_raw /cmd_vel_safe /policy/debug /safety/status \
  /ranger/odom /ranger/status /tf /tf_static
echo "Recorded to ${OUT}"
