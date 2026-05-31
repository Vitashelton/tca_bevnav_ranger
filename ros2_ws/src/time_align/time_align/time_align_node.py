#!/usr/bin/env python3
"""Software time alignment for unsynchronized sensors.

Low-cost platforms cannot hardware-sync the LiDAR, depth camera and
odometry. This node keeps small ring buffers per stream and, on each tick
of the chosen sync anchor, picks the temporally nearest sample from the
other streams. It also computes a per-pair ``time_diff_ms`` and a derived
time confidence ``C_time = exp(-|dt| / tau_t)`` which is forwarded so that
TCA-BEV can downweight poorly-aligned modalities instead of assuming
perfect synchronization.

Outputs:
  /aligned/lidar_points, /aligned/depth_image, /aligned/camera_info,
  /aligned/anchor_odom, /diagnostics/time_align_status (SensorStatus,
  valid_ratio carries the worst time confidence in [0,1]).
"""
import math
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from nav_msgs.msg import Odometry
from e2e_nav_msgs.msg import SensorStatus


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class RingBuffer:
    def __init__(self, size):
        self.buf = deque(maxlen=size)

    def push(self, msg):
        self.buf.append((stamp_to_sec(msg.header.stamp), msg))

    def nearest(self, t):
        if not self.buf:
            return None, None
        best, bdt = None, None
        for (ts, msg) in self.buf:
            dt = abs(ts - t)
            if bdt is None or dt < bdt:
                bdt, best = dt, msg
        return best, bdt


class TimeAlignNode(Node):
    def __init__(self):
        super().__init__('time_align_node')
        self.declare_parameter('sync_anchor', 'lidar')         # lidar|depth|odom
        self.declare_parameter('max_time_diff_ms', 60.0)
        self.declare_parameter('tau_t_ms', 50.0)               # time-confidence scale
        self.declare_parameter('lidar_buffer_size', 10)
        self.declare_parameter('depth_buffer_size', 15)
        self.declare_parameter('odom_buffer_size', 100)
        self.declare_parameter('drop_on_exceed', False)        # drop vs. downweight
        self.declare_parameter('odom_topic', '/mock/odom')

        gp = self.get_parameter
        self.sync_anchor = gp('sync_anchor').value
        self.max_dt = float(gp('max_time_diff_ms').value) * 1e-3
        self.tau_t = float(gp('tau_t_ms').value) * 1e-3
        self.drop_on_exceed = bool(gp('drop_on_exceed').value)
        odom_topic = gp('odom_topic').value

        self.lidar_buf = RingBuffer(int(gp('lidar_buffer_size').value))
        self.depth_buf = RingBuffer(int(gp('depth_buffer_size').value))
        self.cinfo_buf = RingBuffer(int(gp('depth_buffer_size').value))
        self.odom_buf = RingBuffer(int(gp('odom_buffer_size').value))

        self.get_logger().info(
            f"[time_align] anchor={self.sync_anchor} max_dt={self.max_dt*1e3:.1f}ms "
            f"tau_t={self.tau_t*1e3:.1f}ms odom={odom_topic}")

        self.create_subscription(PointCloud2, '/sensors/lidar_points', self.cb_lidar, 5)
        self.create_subscription(Image, '/sensors/depth_image', self.cb_depth, 5)
        self.create_subscription(CameraInfo, '/sensors/camera_info', self.cb_cinfo, 5)
        self.create_subscription(Odometry, odom_topic, self.cb_odom, 20)

        self.pub_lidar = self.create_publisher(PointCloud2, '/aligned/lidar_points', 5)
        self.pub_depth = self.create_publisher(Image, '/aligned/depth_image', 5)
        self.pub_cinfo = self.create_publisher(CameraInfo, '/aligned/camera_info', 5)
        self.pub_odom = self.create_publisher(Odometry, '/aligned/anchor_odom', 10)
        self.pub_status = self.create_publisher(SensorStatus, '/diagnostics/time_align_status', 5)

    def cb_lidar(self, msg):
        self.lidar_buf.push(msg)
        if self.sync_anchor == 'lidar':
            self.align(msg)

    def cb_depth(self, msg):
        self.depth_buf.push(msg)
        if self.sync_anchor == 'depth':
            self.align(msg)

    def cb_cinfo(self, msg):
        self.cinfo_buf.push(msg)

    def cb_odom(self, msg):
        self.odom_buf.push(msg)
        if self.sync_anchor == 'odom':
            self.align(msg)

    def align(self, anchor_msg):
        t = stamp_to_sec(anchor_msg.header.stamp)
        lidar, dl = self.lidar_buf.nearest(t)
        depth, dd = self.depth_buf.nearest(t)
        cinfo, _ = self.cinfo_buf.nearest(t)
        odom, do = self.odom_buf.nearest(t)

        worst_dt = max([d for d in (dl, dd, do) if d is not None], default=0.0)
        c_time = math.exp(-worst_dt / self.tau_t) if self.tau_t > 0 else 1.0

        exceed = worst_dt > self.max_dt
        if exceed and self.drop_on_exceed:
            self._publish_status(worst_dt, c_time, dropped=True)
            return

        if lidar is not None:
            self.pub_lidar.publish(lidar)
        if depth is not None:
            self.pub_depth.publish(depth)
        if cinfo is not None:
            self.pub_cinfo.publish(cinfo)
        if odom is not None:
            self.pub_odom.publish(odom)
        self._publish_status(worst_dt, c_time, dropped=False)

    def _publish_status(self, worst_dt, c_time, dropped):
        st = SensorStatus()
        st.header.stamp = self.get_clock().now().to_msg()
        st.sensor_name = 'time_align'
        st.alive = not dropped
        st.frequency = 0.0
        st.last_time_diff_ms = worst_dt * 1e3
        st.valid_ratio = float(c_time)   # carries time confidence
        st.frame_id = self.sync_anchor
        st.message = 'dropped' if dropped else 'aligned'
        self.pub_status.publish(st)


def main(args=None):
    rclpy.init(args=args)
    node = TimeAlignNode()
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
