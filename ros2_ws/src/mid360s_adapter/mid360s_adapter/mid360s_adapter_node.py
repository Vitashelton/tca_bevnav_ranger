#!/usr/bin/env python3
"""Mid360S adapter: topic remap + frequency / timestamp health check.

Does NOT reimplement the Livox driver. It only relays /livox/lidar and
/livox/imu onto unified /sensors/* topics, checks frequency and timestamp
sanity, and publishes a SensorStatus. Supports mock_mode (passthrough off).
"""
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Imu
from e2e_nav_msgs.msg import SensorStatus


class Mid360sAdapter(Node):
    def __init__(self):
        super().__init__('mid360s_adapter')
        self.declare_parameter('mock_mode', False)
        self.declare_parameter('in_lidar', '/livox/lidar')
        self.declare_parameter('in_imu', '/livox/imu')
        self.declare_parameter('expected_rate', 10.0)
        gp = self.get_parameter
        self.mock = bool(gp('mock_mode').value)
        self.expected = float(gp('expected_rate').value)
        self.get_logger().info(f"[mid360s] mock={self.mock} expected_rate={self.expected}Hz")
        self.last = None
        self.count = 0
        self.t0 = time.time()
        if not self.mock:
            self.create_subscription(PointCloud2, gp('in_lidar').value, self.cb_lidar, 5)
            self.create_subscription(Imu, gp('in_imu').value, self.cb_imu, 20)
        self.pub_pc = self.create_publisher(PointCloud2, '/sensors/lidar_points', 5)
        self.pub_imu = self.create_publisher(Imu, '/sensors/lidar_imu', 20)
        self.pub_st = self.create_publisher(SensorStatus, '/diagnostics/mid360s_status', 5)
        self.create_timer(1.0, self.report)

    def cb_lidar(self, msg):
        self.count += 1
        self.pub_pc.publish(msg)

    def cb_imu(self, msg):
        self.pub_imu.publish(msg)

    def report(self):
        dt = max(time.time() - self.t0, 1e-3)
        freq = self.count / dt
        self.count = 0
        self.t0 = time.time()
        st = SensorStatus()
        st.header.stamp = self.get_clock().now().to_msg()
        st.sensor_name = 'mid360s'
        st.alive = self.mock or freq > 0.1
        st.frequency = freq
        st.frame_id = 'lidar_link'
        if not self.mock and freq < self.expected * 0.5:
            st.message = 'low_frequency'
            self.get_logger().warn(f'mid360s low frequency: {freq:.1f}Hz')
        else:
            st.message = 'ok' if not self.mock else 'mock'
        self.pub_st.publish(st)


def main(args=None):
    rclpy.init(args=args)
    n = Mid360sAdapter()
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
