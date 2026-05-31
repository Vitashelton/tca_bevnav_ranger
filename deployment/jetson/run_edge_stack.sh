#!/usr/bin/env bash
# Launch the full edge navigation stack on the Jetson with real sensors.
set -e
WS="$(cd "$(dirname "$0")/../../ros2_ws" && pwd)"
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
ANCHOR_BACKEND="${ANCHOR_BACKEND:-wheel_odom}"   # none|wheel_odom|fast_lio2|fast_livo2
RUNTIME="${RUNTIME:-tensorrt}"                    # tensorrt|onnx|torch|dummy
exec ros2 launch tca_bev_bringup jetson_edge.launch.py \
  anchor_backend:="${ANCHOR_BACKEND}" runtime_type:="${RUNTIME}"
