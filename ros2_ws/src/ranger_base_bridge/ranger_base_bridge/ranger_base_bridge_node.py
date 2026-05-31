#!/usr/bin/env python3
"""Ranger Mini 2.0 base bridge.

Execution interface only -- it does NOT do any path planning. Subscribes
to /cmd_vel_safe and drives the omni-directional base (vx, vy, wz). In mock
mode it integrates a simple kinematic model and publishes a synthetic
/ranger/odom so the rest of the pipeline can be exercised without hardware.

Real-robot mode is left as a clearly marked TODO: connect to the AgileX
ranger_ros2 driver or an existing CAN driver. The topic interface
(/cmd_vel_safe in, /ranger/odom + /ranger/status out) is fixed so swapping
the backend does not affect the rest of the stack.
"""
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import math


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class RangerBaseBridge(Node):
    def __init__(self):
        super().__init__('ranger_base_bridge')
        self.declare_parameter('mock_mode', True)
        self.declare_parameter('max_vx', 1.0)
        self.declare_parameter('max_vy', 0.6)
        self.declare_parameter('max_wz', 1.2)
        self.declare_parameter('cmd_timeout', 0.5)
        self.declare_parameter('odom_rate', 50.0)
        self.declare_parameter('publish_tf', False)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        gp = self.get_parameter
        self.mock = bool(gp('mock_mode').value)
        self.max_vx = float(gp('max_vx').value)
        self.max_vy = float(gp('max_vy').value)
        self.max_wz = float(gp('max_wz').value)
        self.cmd_timeout = float(gp('cmd_timeout').value)
        self.publish_tf = bool(gp('publish_tf').value)
        self.odom_frame = gp('odom_frame').value
        self.base_frame = gp('base_frame').value

        self.get_logger().info(
            f"[ranger_base] mock_mode={self.mock} "
            f"max=({self.max_vx},{self.max_vy},{self.max_wz})")

        if not self.mock:
            # TODO(real-robot): initialize AgileX ranger_ros2 / CAN driver here.
            # e.g. self.driver = RangerDriver(can_port='can0')
            self.get_logger().warn(
                'real-robot mode requested but driver is not implemented. '
                'TODO: wire ranger_ros2 / CAN. Running as passive bridge.')

        self.cmd = (0.0, 0.0, 0.0)
        self.last_cmd_t = 0.0
        self.x = self.y = self.yaw = 0.0
        self.last_t = time.time()

        self.create_subscription(Twist, '/cmd_vel_safe', self.cb_cmd, 5)
        self.pub_odom = self.create_publisher(Odometry, '/ranger/odom', 10)
        self.pub_status = self.create_publisher(String, '/ranger/status', 5)
        if self.publish_tf:
            from tf2_ros import TransformBroadcaster
            self.tf_bc = TransformBroadcaster(self)
        self.create_timer(1.0 / float(gp('odom_rate').value), self.tick)

    def cb_cmd(self, msg):
        vx = max(-self.max_vx, min(self.max_vx, msg.linear.x))
        vy = max(-self.max_vy, min(self.max_vy, msg.linear.y))
        wz = max(-self.max_wz, min(self.max_wz, msg.angular.z))
        self.cmd = (vx, vy, wz)
        self.last_cmd_t = time.time()
        if not self.mock:
            # TODO(real-robot): self.driver.set_velocity(vx, vy, wz)
            pass

    def tick(self):
        now = time.time()
        dt = now - self.last_t
        self.last_t = now
        vx, vy, wz = self.cmd
        if (now - self.last_cmd_t) > self.cmd_timeout:
            vx = vy = wz = 0.0          # safety: zero on stale command

        # integrate body velocity into odom (mock kinematic model)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (c * vx - s * vy) * dt
        self.y += (s * vx + c * vy) * dt
        self.yaw += wz * dt

        od = Odometry()
        od.header.stamp = self.get_clock().now().to_msg()
        od.header.frame_id = self.odom_frame
        od.child_frame_id = self.base_frame
        od.pose.pose.position.x = self.x
        od.pose.pose.position.y = self.y
        qx, qy, qz, qw = yaw_to_quat(self.yaw)
        od.pose.pose.orientation.x = qx
        od.pose.pose.orientation.y = qy
        od.pose.pose.orientation.z = qz
        od.pose.pose.orientation.w = qw
        od.twist.twist.linear.x = vx
        od.twist.twist.linear.y = vy
        od.twist.twist.angular.z = wz
        self.pub_odom.publish(od)

        st = String()
        st.data = f"mock={self.mock} cmd=({vx:.2f},{vy:.2f},{wz:.2f})"
        self.pub_status.publish(st)

        if self.publish_tf:
            t = TransformStamped()
            t.header = od.header
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_bc.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = RangerBaseBridge()
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
