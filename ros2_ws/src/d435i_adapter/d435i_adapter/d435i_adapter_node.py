#!/usr/bin/env python3
"""RealSense D435i adapter: depth/color/camera_info relay + valid ratio.

Does NOT reimplement realsense2_camera. Relays raw topics to /sensors/*,
applies depth_scale and min/max clipping, optional downsample, and reports
valid_depth_ratio. Supports mock_mode (passthrough off).
"""
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from e2e_nav_msgs.msg import SensorStatus


class D435iAdapter(Node):
    def __init__(self):
        super().__init__('d435i_adapter')
        self.declare_parameter('mock_mode', False)
        self.declare_parameter('in_depth', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('in_color', '/camera/color/image_raw')
        self.declare_parameter('in_cinfo', '/camera/color/camera_info')
        self.declare_parameter('depth_scale', 0.001)   # 16UC1 mm -> m
        self.declare_parameter('min_depth', 0.2)
        self.declare_parameter('max_depth', 6.0)
        self.declare_parameter('downsample', 1)
        gp = self.get_parameter
        self.mock = bool(gp('mock_mode').value)
        self.scale = float(gp('depth_scale').value)
        self.min_d = float(gp('min_depth').value)
        self.max_d = float(gp('max_depth').value)
        self.ds = int(gp('downsample').value)
        self.get_logger().info(f"[d435i] mock={self.mock} scale={self.scale} ds={self.ds}")
        if not self.mock:
            self.create_subscription(Image, gp('in_depth').value, self.cb_depth, 5)
            self.create_subscription(Image, gp('in_color').value, self.cb_color, 5)
            self.create_subscription(CameraInfo, gp('in_cinfo').value, self.cb_cinfo, 5)
        self.pub_depth = self.create_publisher(Image, '/sensors/depth_image', 5)
        self.pub_color = self.create_publisher(Image, '/sensors/color_image', 5)
        self.pub_cinfo = self.create_publisher(CameraInfo, '/sensors/camera_info', 5)
        self.pub_st = self.create_publisher(SensorStatus, '/diagnostics/d435i_status', 5)

    def cb_color(self, msg):
        self.pub_color.publish(msg)

    def cb_cinfo(self, msg):
        self.pub_cinfo.publish(msg)

    def cb_depth(self, msg):
        try:
            if msg.encoding in ('16UC1', 'mono16'):
                d = np.frombuffer(bytes(msg.data), np.uint16).reshape(
                    msg.height, msg.width).astype(np.float32) * self.scale
            elif msg.encoding == '32FC1':
                d = np.frombuffer(bytes(msg.data), np.float32).reshape(
                    msg.height, msg.width).copy()
            else:
                self.get_logger().warn(f'unsupported depth encoding {msg.encoding}')
                return
        except Exception as e:
            self.get_logger().warn(f'depth decode failed: {e}')
            return
        d[(d < self.min_d) | (d > self.max_d)] = np.nan
        if self.ds > 1:
            d = d[::self.ds, ::self.ds]
        valid = float(np.isfinite(d).mean())
        out = Image()
        out.header = msg.header
        out.height, out.width = d.shape
        out.encoding = '32FC1'
        out.is_bigendian = 0
        out.step = d.shape[1] * 4
        out.data = d.astype(np.float32).tobytes()
        self.pub_depth.publish(out)
        st = SensorStatus()
        st.header.stamp = self.get_clock().now().to_msg()
        st.sensor_name = 'd435i'
        st.alive = True
        st.valid_ratio = valid
        st.frame_id = msg.header.frame_id
        st.message = 'ok'
        self.pub_st.publish(st)


def main(args=None):
    rclpy.init(args=args)
    n = D435iAdapter()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
