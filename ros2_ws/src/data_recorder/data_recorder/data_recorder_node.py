#!/usr/bin/env python3
"""Dataset recorder helper.

Thin wrapper that writes a metadata.yaml for a recording session. The
actual bag recording is done by `ros2 bag record` (see scripts/record_all.sh
and data_recorder/launch). This node just timestamps and labels a session so
datasets are self-describing for training / evaluation.
"""
import os
import datetime
import rclpy
from rclpy.node import Node
import yaml


class DataRecorder(Node):
    def __init__(self):
        super().__init__('data_recorder')
        self.declare_parameter('output_dir', os.path.expanduser('~/tca_datasets'))
        self.declare_parameter('scenario_name', 'unnamed')
        self.declare_parameter('dynamic_obstacles', 0)
        self.declare_parameter('door_count', 0)
        self.declare_parameter('corridor_width', 0.0)
        self.declare_parameter('anchor_backend', 'mock_odom')
        self.declare_parameter('policy_version', 'dummy')
        self.declare_parameter('operator', 'unknown')
        gp = self.get_parameter
        out = os.path.join(gp('output_dir').value,
                           gp('scenario_name').value + '_' +
                           datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(out, exist_ok=True)
        meta = {
            'scenario_name': gp('scenario_name').value,
            'date': datetime.datetime.now().isoformat(),
            'dynamic_obstacles': int(gp('dynamic_obstacles').value),
            'door_count': int(gp('door_count').value),
            'corridor_width': float(gp('corridor_width').value),
            'anchor_backend': gp('anchor_backend').value,
            'policy_version': gp('policy_version').value,
            'operator': gp('operator').value,
            'sensor_config': {'lidar': 'Mid360S', 'camera': 'D435i'},
        }
        with open(os.path.join(out, 'metadata.yaml'), 'w') as f:
            yaml.safe_dump(meta, f, allow_unicode=True)
        self.get_logger().info(f"[recorder] metadata written to {out}/metadata.yaml")
        self.get_logger().info("Run scripts/record_all.sh to start `ros2 bag record`.")


def main(args=None):
    rclpy.init(args=args)
    n = DataRecorder()
    try:
        rclpy.spin_once(n, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
