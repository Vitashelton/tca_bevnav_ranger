#!/usr/bin/env python3
"""Calibration uncertainty manager for the weakly-calibrated regime.

Mid360S<->D435i extrinsics are not yet precisely calibrated. Instead of
pretending the transforms are exact, this node:
  * holds the (rough, hand-measured) base_link->lidar_link and
    base_link->camera_link extrinsics from configuration,
  * exposes per-sensor *extrinsic confidence* values,
  * derives a depth obstacle inflation radius that grows when calibration
    confidence is low (conservative mode), and
  * publishes everything on /calibration/status so TCA-BEV can query it.

It also broadcasts the rough extrinsics on /tf_static so the rest of the
stack has *some* transform to use, clearly flagged as low-confidence.

C_calib = exp(-sigma_ext / tau_c)  (see docs/tca_bev_method.md)
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from e2e_nav_msgs.msg import SensorStatus


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class CalibrationUncertaintyManager(Node):
    def __init__(self):
        super().__init__('calibration_uncertainty_manager')
        # rough extrinsics base_link -> lidar_link
        self.declare_parameter('lidar_xyz', [0.0, 0.0, 0.30])
        self.declare_parameter('lidar_yaw', 0.0)
        # rough extrinsics base_link -> camera_link
        self.declare_parameter('camera_xyz', [0.20, 0.0, 0.20])
        self.declare_parameter('camera_yaw', 0.0)
        # confidences in [0,1]
        self.declare_parameter('lidar_extrinsic_confidence', 0.9)
        self.declare_parameter('camera_extrinsic_confidence', 0.4)
        # depth inflation radius (m) interpolated between low/high confidence
        self.declare_parameter('depth_inflation_radius_low_conf', 0.40)
        self.declare_parameter('depth_inflation_radius_high_conf', 0.10)
        self.declare_parameter('tau_c', 0.5)
        self.declare_parameter('broadcast_tf', True)
        self.declare_parameter('publish_rate', 2.0)

        gp = self.get_parameter
        self.lidar_xyz = list(gp('lidar_xyz').value)
        self.lidar_yaw = float(gp('lidar_yaw').value)
        self.camera_xyz = list(gp('camera_xyz').value)
        self.camera_yaw = float(gp('camera_yaw').value)
        self.lidar_conf = float(gp('lidar_extrinsic_confidence').value)
        self.camera_conf = float(gp('camera_extrinsic_confidence').value)
        self.infl_low = float(gp('depth_inflation_radius_low_conf').value)
        self.infl_high = float(gp('depth_inflation_radius_high_conf').value)
        self.tau_c = float(gp('tau_c').value)
        self.broadcast_tf = bool(gp('broadcast_tf').value)

        self.get_logger().info(
            f"[calib] lidar_conf={self.lidar_conf} camera_conf={self.camera_conf} "
            f"depth_inflation=[{self.infl_high},{self.infl_low}]m")

        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.pub_status = self.create_publisher(SensorStatus, '/calibration/status', qos)

        if self.broadcast_tf:
            self.tf_static = StaticTransformBroadcaster(self)
            self._broadcast_static()

        self.create_timer(1.0 / float(gp('publish_rate').value), self.tick)

    def depth_inflation_radius(self):
        """Linear interpolation: lower camera confidence -> larger inflation."""
        c = max(0.0, min(1.0, self.camera_conf))
        return self.infl_low + (self.infl_high - self.infl_low) * c

    def calib_confidence(self):
        """C_calib = exp(-sigma_ext / tau_c); sigma_ext ~ (1 - camera_conf)."""
        sigma_ext = 1.0 - self.camera_conf
        return math.exp(-sigma_ext / self.tau_c) if self.tau_c > 0 else self.camera_conf

    def _broadcast_static(self):
        tfs = []
        now = self.get_clock().now().to_msg()
        for child, xyz, yaw in (
            ('lidar_link', self.lidar_xyz, self.lidar_yaw),
            ('camera_link', self.camera_xyz, self.camera_yaw),
        ):
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = 'base_link'
            t.child_frame_id = child
            t.transform.translation.x = float(xyz[0])
            t.transform.translation.y = float(xyz[1])
            t.transform.translation.z = float(xyz[2])
            qx, qy, qz, qw = yaw_to_quat(yaw)
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            tfs.append(t)
        # camera_link -> camera_depth_optical_frame (REP-103 optical convention)
        opt = TransformStamped()
        opt.header.stamp = now
        opt.header.frame_id = 'camera_link'
        opt.child_frame_id = 'camera_depth_optical_frame'
        # optical: x-right, y-down, z-forward  => rotation from body frame
        opt.transform.rotation.x = -0.5
        opt.transform.rotation.y = 0.5
        opt.transform.rotation.z = -0.5
        opt.transform.rotation.w = 0.5
        tfs.append(opt)
        self.tf_static.sendTransform(tfs)

    def tick(self):
        c_calib = self.calib_confidence()
        st = SensorStatus()
        st.header.stamp = self.get_clock().now().to_msg()
        st.sensor_name = 'calibration'
        st.alive = True
        # encode: frequency<-lidar_conf, valid_ratio<-camera C_calib,
        # last_time_diff_ms<-depth_inflation_radius(m)*1000
        st.frequency = float(self.lidar_conf)
        st.valid_ratio = float(c_calib)
        st.last_time_diff_ms = float(self.depth_inflation_radius() * 1000.0)
        st.frame_id = 'base_link'
        st.message = (f"lidar_conf={self.lidar_conf:.2f} camera_C_calib={c_calib:.2f} "
                      f"depth_infl={self.depth_inflation_radius():.2f}m")
        self.pub_status.publish(st)


def main(args=None):
    rclpy.init(args=args)
    node = CalibrationUncertaintyManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
