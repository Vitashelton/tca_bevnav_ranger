#!/usr/bin/env bash
# Run the mock closed loop on the Jetson (no hardware) -- useful for smoke tests.
set -e
WS="$(cd "$(dirname "$0")/../../ros2_ws" && pwd)"
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
exec ros2 launch tca_bev_bringup mock_e2e_closed_loop.launch.py \
  scenario:="${SCENARIO:-corridor_door}" fusion_mode:="${FUSION:-tca}"
