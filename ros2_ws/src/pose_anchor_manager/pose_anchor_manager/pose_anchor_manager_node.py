#!/usr/bin/env python3
"""State-estimation-agnostic pose anchor manager.

This node does NOT implement FAST-LIO / FAST-LIVO. It is a thin adapter
that subscribes to whatever odometry topic a chosen *backend* produces,
re-publishes it on a unified ``/anchor/odom`` topic, and estimates an
``anchor_quality`` in [0,1] from frequency, pose-jump, velocity
consistency and covariance. The anchor quality is consumed by TCA-BEV and
the policy so that a degraded state estimator triggers conservative
behavior instead of silent failure.

backend_type -> default input topic (override via 'input_topic'):
  none / mock_odom -> /mock/odom
  wheel_odom       -> /ranger/odom
  fast_lio2        -> /Odometry
  fast_livo        -> /fast_livo/odom
  fast_livo2       -> /fast_livo/odom
  external_odom    -> /external/odom
"""
import math
from collections import deque

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from e2e_nav_msgs.msg import AnchorStatus


def stamp_to_sec(s):
    return s.sec + s.nanosec * 1e-9


DEFAULT_TOPIC = {
    'none': '/mock/odom',
    'mock_odom': '/mock/odom',
    'wheel_odom': '/ranger/odom',
    'fast_lio2': '/Odometry',
    'fast_livo': '/fast_livo/odom',
    'fast_livo2': '/fast_livo/odom',
    'external_odom': '/external/odom',
}


class PoseAnchorManager(Node):
    def __init__(self):
        super().__init__('pose_anchor_manager')
        self.declare_parameter('backend_type', 'mock_odom')
        self.declare_parameter('input_topic', '')           # '' -> use default for backend
        self.declare_parameter('expected_freq', 50.0)
        self.declare_parameter('tau_jump', 0.3)             # m per step scale
        self.declare_parameter('tau_vel', 0.5)              # m/s inconsistency scale
        self.declare_parameter('tau_cov', 0.5)              # covariance norm scale
        self.declare_parameter('min_quality_valid', 0.15)
        self.declare_parameter('path_max_len', 500)

        gp = self.get_parameter
        self.backend = gp('backend_type').value
        topic = gp('input_topic').value or DEFAULT_TOPIC.get(self.backend, '/mock/odom')
        self.expected_freq = float(gp('expected_freq').value)
        self.tau_jump = float(gp('tau_jump').value)
        self.tau_vel = float(gp('tau_vel').value)
        self.tau_cov = float(gp('tau_cov').value)
        self.min_quality_valid = float(gp('min_quality_valid').value)
        self.path_max_len = int(gp('path_max_len').value)

        self.get_logger().info(
            f"[pose_anchor] backend={self.backend} input_topic={topic} "
            f"expected_freq={self.expected_freq}Hz")

        self.times = deque(maxlen=30)
        self.last_pose = None       # (t, x, y, yaw)
        self.last_vel = None
        self.path = Path()

        self.create_subscription(Odometry, topic, self.cb_odom, 20)
        self.pub_odom = self.create_publisher(Odometry, '/anchor/odom', 10)
        self.pub_path = self.create_publisher(Path, '/anchor/path', 1)
        self.pub_status = self.create_publisher(AnchorStatus, '/anchor/status', 5)

        # watchdog: if no input, publish a degraded status periodically
        self.last_msg_time = None
        self.create_timer(0.5, self.watchdog)

    @staticmethod
    def _yaw(q):
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def cb_odom(self, msg):
        now = stamp_to_sec(msg.header.stamp)
        if now <= 0.0:
            now = self.get_clock().now().nanoseconds * 1e-9
        self.last_msg_time = self.get_clock().now()
        self.times.append(now)

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self._yaw(msg.pose.pose.orientation)

        # frequency
        freq = 0.0
        if len(self.times) >= 2:
            dt_total = self.times[-1] - self.times[0]
            if dt_total > 1e-6:
                freq = (len(self.times) - 1) / dt_total

        # pose jump + velocity from measured displacement
        jump = 0.0
        meas_vx = meas_vy = 0.0
        if self.last_pose is not None:
            lt, lx, ly, _ = self.last_pose
            dt = max(now - lt, 1e-3)
            dx, dy = x - lx, y - ly
            disp = math.hypot(dx, dy)
            jump = disp
            meas_vx, meas_vy = dx / dt, dy / dt
        self.last_pose = (now, x, y, yaw)

        # velocity consistency vs. reported twist (if any)
        rep_vx = msg.twist.twist.linear.x
        rep_vy = msg.twist.twist.linear.y
        vel_inconsistency = math.hypot(meas_vx - rep_vx, meas_vy - rep_vy)

        # covariance norm (position part)
        cov = msg.pose.covariance
        cov_norm = math.sqrt(max(cov[0], 0.0) + max(cov[7], 0.0) + max(cov[35], 0.0)) \
            if len(cov) >= 36 else 0.0

        q_freq = max(0.0, min(1.0, freq / self.expected_freq)) if self.expected_freq > 0 else 1.0
        q_jump = math.exp(-jump / self.tau_jump) if self.tau_jump > 0 else 1.0
        q_vel = math.exp(-vel_inconsistency / self.tau_vel) if self.tau_vel > 0 else 1.0
        q_cov = math.exp(-cov_norm / self.tau_cov) if self.tau_cov > 0 else 1.0
        quality = q_freq * q_jump * q_vel * q_cov

        is_valid = quality >= self.min_quality_valid and self.backend != 'none'
        degraded = quality < 0.5

        # republish unified anchor odom
        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id or 'base_link'
        out.pose = msg.pose
        out.twist = msg.twist
        self.pub_odom.publish(out)

        # path
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose
        self.path.header = msg.header
        self.path.poses.append(ps)
        if len(self.path.poses) > self.path_max_len:
            self.path.poses = self.path.poses[-self.path_max_len:]
        self.pub_path.publish(self.path)

        self._publish_status(is_valid, quality, freq, jump,
                             max(0.0, 1.0 - vel_inconsistency), cov_norm, degraded,
                             'ok' if is_valid else 'low_quality')

    def watchdog(self):
        if self.last_msg_time is None:
            self._publish_status(False, 0.0, 0.0, 0.0, 0.0, 0.0, True, 'no_input')
            return
        age = (self.get_clock().now() - self.last_msg_time).nanoseconds * 1e-9
        if age > 1.0:
            self._publish_status(False, 0.0, 0.0, 0.0, 0.0, 0.0, True, 'stale_input')

    def _publish_status(self, valid, quality, freq, jump, vel_cons, cov, degraded, msg):
        st = AnchorStatus()
        st.header.stamp = self.get_clock().now().to_msg()
        st.backend_type = self.backend
        st.is_valid = bool(valid)
        st.anchor_quality = float(quality)
        st.odom_frequency = float(freq)
        st.pose_jump_score = float(jump)
        st.velocity_consistency = float(vel_cons)
        st.covariance_score = float(cov)
        st.degraded_mode = bool(degraded)
        st.message = msg
        self.pub_status.publish(st)


def main(args=None):
    rclpy.init(args=args)
    node = PoseAnchorManager()
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
