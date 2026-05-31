#!/usr/bin/env python3
"""Independent safety supervisor (mandatory on the real robot).

Sits between the policy (/cmd_vel_raw) and the base (/cmd_vel_safe). The
policy can never bypass it. Responsibilities:
  * highest-priority E-stop (latched until cleared),
  * minimum-obstacle-distance gating from the BEV occupancy,
  * forward safety-zone check,
  * velocity clamping,
  * input-timeout protection (zero command if BEV / cmd are stale),
  * degraded-mode and uncertainty-based speed scaling.

Any critical-input timeout => zero velocity. This is intentionally simple
and reactive; it does not depend on the learned policy being correct.
"""
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from e2e_nav_msgs.msg import BevTensor, SafetyStatus, PolicyDebug, AnchorStatus


class SafetySupervisor(Node):
    def __init__(self):
        super().__init__('safety_supervisor')
        self.declare_parameter('max_vx', 0.6)
        self.declare_parameter('max_vy', 0.4)
        self.declare_parameter('max_wz', 1.0)
        self.declare_parameter('min_obstacle_distance', 0.30)
        self.declare_parameter('slow_obstacle_distance', 0.8)
        self.declare_parameter('cmd_timeout', 0.5)
        self.declare_parameter('bev_timeout', 0.5)
        self.declare_parameter('degraded_speed_scale', 0.4)
        self.declare_parameter('uncertainty_brake', True)
        self.declare_parameter('output_rate', 20.0)

        gp = self.get_parameter
        self.max_vx = float(gp('max_vx').value)
        self.max_vy = float(gp('max_vy').value)
        self.max_wz = float(gp('max_wz').value)
        self.min_dist = float(gp('min_obstacle_distance').value)
        self.slow_dist = float(gp('slow_obstacle_distance').value)
        self.cmd_timeout = float(gp('cmd_timeout').value)
        self.bev_timeout = float(gp('bev_timeout').value)
        self.degraded_scale = float(gp('degraded_speed_scale').value)
        self.unc_brake = bool(gp('uncertainty_brake').value)

        self.get_logger().info(
            f"[safety] min_dist={self.min_dist} slow_dist={self.slow_dist} "
            f"cmd_timeout={self.cmd_timeout}s bev_timeout={self.bev_timeout}s")

        self.raw = (0.0, 0.0, 0.0)
        self.last_cmd_t = 0.0
        self.bev = None
        self.last_bev_t = 0.0
        self.estop = False
        self.uncertainty = 0.0
        self.degraded = False

        self.create_subscription(Twist, '/cmd_vel_raw', self.cb_cmd, 5)
        self.create_subscription(BevTensor, '/bev/tensor', self.cb_bev, 5)
        self.create_subscription(PolicyDebug, '/policy/debug', self.cb_dbg, 5)
        self.create_subscription(AnchorStatus, '/anchor/status', self.cb_anchor, 5)
        self.create_subscription(Bool, '/manual_estop', self.cb_estop, 1)

        self.pub_safe = self.create_publisher(Twist, '/cmd_vel_safe', 5)
        self.pub_status = self.create_publisher(SafetyStatus, '/safety/status', 5)
        self.create_timer(1.0 / float(gp('output_rate').value), self.tick)

    def cb_cmd(self, msg):
        self.raw = (msg.linear.x, msg.linear.y, msg.angular.z)
        self.last_cmd_t = time.time()

    def cb_bev(self, msg):
        self.bev = msg
        self.last_bev_t = time.time()

    def cb_dbg(self, msg):
        self.uncertainty = float(msg.uncertainty)

    def cb_anchor(self, msg):
        self.degraded = bool(msg.degraded_mode) or not msg.is_valid

    def cb_estop(self, msg):
        if msg.data:
            self.estop = True
            self.get_logger().warn('E-STOP engaged')
        else:
            self.estop = False
            self.get_logger().info('E-STOP cleared')

    def _forward_min_dist(self):
        """Minimum obstacle distance in a forward cone from the BEV."""
        if self.bev is None:
            return None
        c, h, w = self.bev.channels, self.bev.height, self.bev.width
        if len(self.bev.data) < h * w:
            return None
        occ = np.array(self.bev.data[:h * w], np.float32).reshape(h, w)
        res = self.bev.resolution
        cx, cy = h // 2, w // 2
        best = self.slow_dist + 1.0
        for ang in np.linspace(-0.6, 0.6, 13):
            for r in np.arange(0.05, self.slow_dist + 1.0, res):
                i = int(cx + (r / res) * math.cos(ang))
                j = int(cy + (r / res) * math.sin(ang))
                if 0 <= i < h and 0 <= j < w and occ[i, j] > 0.5:
                    best = min(best, r)
                    break
        return best

    def tick(self):
        now = time.time()
        vx, vy, wz = self.raw
        status = SafetyStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        reason = 'ok'
        safe = True
        too_close = False
        timeout = False

        # 1) E-stop wins
        if self.estop:
            self._emit(0, 0, 0, status, False, True, False, False, 99.0, 'estop')
            return

        # 2) input timeout
        if (now - self.last_cmd_t) > self.cmd_timeout or self.last_cmd_t == 0.0:
            timeout = True
            reason = 'cmd_timeout'
        if (now - self.last_bev_t) > self.bev_timeout or self.last_bev_t == 0.0:
            timeout = True
            reason = 'bev_timeout'
        if timeout:
            self._emit(0, 0, 0, status, False, False, False, True, 99.0, reason)
            return

        # 3) obstacle gating
        fmin = self._forward_min_dist()
        if fmin is None:
            fmin = 99.0
        if fmin < self.min_dist and vx > 0.0:
            vx = 0.0
            too_close = True
            safe = False
            reason = 'obstacle_too_close'
        elif fmin < self.slow_dist and vx > 0.0:
            scale = (fmin - self.min_dist) / max(self.slow_dist - self.min_dist, 1e-3)
            vx *= float(np.clip(scale, 0.0, 1.0))
            reason = 'slow_zone'

        # 4) degraded / uncertainty scaling
        if self.degraded:
            vx *= self.degraded_scale
            vy *= self.degraded_scale
            wz *= self.degraded_scale
            reason = reason + '+degraded'
        if self.unc_brake and self.uncertainty > 0.5:
            f = float(np.clip(1.0 - (self.uncertainty - 0.5) * 2.0, 0.1, 1.0))
            vx *= f; vy *= f; wz *= f
            reason = reason + '+uncertain'

        # 5) clamp
        vx = float(np.clip(vx, -self.max_vx, self.max_vx))
        vy = float(np.clip(vy, -self.max_vy, self.max_vy))
        wz = float(np.clip(wz, -self.max_wz, self.max_wz))

        self._emit(vx, vy, wz, status, safe, False, too_close, False, fmin, reason)

    def _emit(self, vx, vy, wz, status, safe, estop, too_close, timeout, fmin, reason):
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = float(vx), float(vy), float(wz)
        self.pub_safe.publish(cmd)
        status.safe = bool(safe)
        status.estop = bool(estop)
        status.obstacle_too_close = bool(too_close)
        status.input_timeout = bool(timeout)
        status.min_obstacle_distance = float(fmin)
        status.vx_limited = float(vx)
        status.vy_limited = float(vy)
        status.wz_limited = float(wz)
        status.reason = reason
        self.pub_status.publish(status)


def main(args=None):
    rclpy.init(args=args)
    node = SafetySupervisor()
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
