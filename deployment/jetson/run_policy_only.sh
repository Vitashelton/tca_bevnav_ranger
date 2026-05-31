#!/usr/bin/env bash
# Run just the policy runtime + safety supervisor (sensors/BEV assumed running).
set -e
WS="$(cd "$(dirname "$0")/../../ros2_ws" && pwd)"
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
exec ros2 launch tca_bev_bringup policy_runtime.launch.py \
  runtime_type:="${RUNTIME:-tensorrt}"
