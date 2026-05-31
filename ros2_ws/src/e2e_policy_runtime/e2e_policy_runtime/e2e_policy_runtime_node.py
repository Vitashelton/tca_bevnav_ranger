#!/usr/bin/env python3
"""End-to-end policy runtime.

Loads a policy that maps (BEV tensor, goal vector, current velocity,
anchor quality) -> (vx, vy, wz, uncertainty) and publishes the *raw*
command on /cmd_vel_raw. The raw command MUST pass through the safety
supervisor before reaching the base; this node never publishes to the
base directly.

Runtimes:
  dummy     -> hand-crafted reactive controller (no learning, always works)
  torch     -> TODO: load a TorchScript / nn.Module checkpoint
  onnx      -> TODO: onnxruntime.InferenceSession
  tensorrt  -> TODO: deployment.tensorrt.infer_trt engine

Stage 1 ships the dummy runtime fully; learned runtimes expose a clear
adapter point (self._infer_*).
"""
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from e2e_nav_msgs.msg import BevTensor, GoalVector, PolicyDebug


class PolicyRuntime(Node):
    def __init__(self):
        super().__init__('e2e_policy_runtime')
        self.declare_parameter('runtime_type', 'dummy')     # dummy|torch|onnx|tensorrt
        self.declare_parameter('model_path', '')
        self.declare_parameter('max_vx', 0.6)
        self.declare_parameter('max_vy', 0.4)
        self.declare_parameter('max_wz', 1.0)
        self.declare_parameter('goal_gain', 1.0)
        self.declare_parameter('obstacle_brake_dist', 0.8)
        self.declare_parameter('require_enable', False)
        self.declare_parameter('control_rate', 10.0)

        gp = self.get_parameter
        self.runtime_type = gp('runtime_type').value
        self.model_path = gp('model_path').value
        self.max_vx = float(gp('max_vx').value)
        self.max_vy = float(gp('max_vy').value)
        self.max_wz = float(gp('max_wz').value)
        self.goal_gain = float(gp('goal_gain').value)
        self.brake_dist = float(gp('obstacle_brake_dist').value)
        self.require_enable = bool(gp('require_enable').value)

        self.get_logger().info(
            f"[policy] runtime={self.runtime_type} model='{self.model_path}' "
            f"max=({self.max_vx},{self.max_vy},{self.max_wz})")

        self.bev = None
        self.goal = None
        self.vel = (0.0, 0.0, 0.0)
        self.enabled = not self.require_enable
        self.session = None
        self._load_runtime()

        self.create_subscription(BevTensor, '/bev/tensor', self.cb_bev, 5)
        self.create_subscription(GoalVector, '/goal/vector', self.cb_goal, 5)
        self.create_subscription(Odometry, '/anchor/odom', self.cb_odom, 10)
        self.create_subscription(Bool, '/manual_enable', self.cb_enable, 1)

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel_raw', 5)
        self.pub_dbg = self.create_publisher(PolicyDebug, '/policy/debug', 5)
        self.create_timer(1.0 / float(gp('control_rate').value), self.tick)

    def _load_runtime(self):
        if self.runtime_type == 'dummy':
            return
        if self.runtime_type == 'torch':
            # TODO: import torch; self.session = torch.jit.load(self.model_path).eval()
            self.get_logger().warn('torch runtime not wired; falling back to dummy. '
                                   'TODO: load TorchScript at self._infer_torch')
            self.runtime_type = 'dummy'
        elif self.runtime_type == 'onnx':
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(self.model_path)
            except Exception as e:
                self.get_logger().warn(f'onnx load failed ({e}); using dummy')
                self.runtime_type = 'dummy'
        elif self.runtime_type == 'tensorrt':
            # TODO: from deployment.tensorrt.infer_trt import TrtPolicy
            self.get_logger().warn('tensorrt runtime not wired; using dummy. '
                                   'TODO: load engine at self._infer_trt')
            self.runtime_type = 'dummy'

    def cb_bev(self, msg): self.bev = msg
    def cb_goal(self, msg): self.goal = msg
    def cb_enable(self, msg): self.enabled = bool(msg.data)

    def cb_odom(self, msg):
        self.vel = (msg.twist.twist.linear.x, msg.twist.twist.linear.y,
                    msg.twist.twist.angular.z)

    # ---- BEV decode ----
    def _bev_array(self):
        if self.bev is None:
            return None
        c, h, w = self.bev.channels, self.bev.height, self.bev.width
        if len(self.bev.data) != c * h * w:
            return None
        return np.array(self.bev.data, np.float32).reshape(c, h, w)

    # ---- DummyPolicy: reactive omni-directional controller ----
    def _infer_dummy(self, bev, goal):
        if goal is None or not goal.has_goal:
            return 0.0, 0.0, 0.0, 0.1
        h, w = bev.shape[1], bev.shape[2]
        res = self.bev.resolution
        occ = bev[0]                      # fused occupancy (ch0)
        cx, cy = h // 2, w // 2           # robot cell

        # heading toward goal
        heading = goal.heading_error
        speed = min(self.max_vx, self.goal_gain * goal.distance)

        # forward obstacle proximity: scan a forward cone
        fwd_min = self._cone_min_dist(occ, cx, cy, res, ang0=-0.5, ang1=0.5, rmax=self.brake_dist)
        left_free = self._cone_min_dist(occ, cx, cy, res, ang0=0.5, ang1=1.4, rmax=2.0)
        right_free = self._cone_min_dist(occ, cx, cy, res, ang0=-1.4, ang1=-0.5, rmax=2.0)

        brake = np.clip(fwd_min / max(self.brake_dist, 1e-3), 0.0, 1.0)
        vx = speed * math.cos(heading) * brake
        vy = speed * math.sin(heading)

        # lateral escape using omni vy if forward is blocked
        if fwd_min < 0.5:
            if left_free > right_free:
                vy += 0.5 * self.max_vy
            else:
                vy -= 0.5 * self.max_vy
        wz = np.clip(1.5 * heading, -self.max_wz, self.max_wz)

        vx = float(np.clip(vx, -self.max_vx, self.max_vx))
        vy = float(np.clip(vy, -self.max_vy, self.max_vy))
        unc = float(np.clip(1.0 - self.bev.global_confidence, 0.0, 1.0))
        return vx, vy, float(wz), unc

    @staticmethod
    def _cone_min_dist(occ, cx, cy, res, ang0, ang1, rmax):
        best = rmax
        for ang in np.linspace(ang0, ang1, 9):
            for r in np.arange(0.1, rmax, res * 2):
                i = int(cx + (r / res) * math.cos(ang))
                j = int(cy + (r / res) * math.sin(ang))
                if 0 <= i < occ.shape[0] and 0 <= j < occ.shape[1]:
                    if occ[i, j] > 0.5:
                        best = min(best, r)
                        break
        return best

    def _infer_onnx(self, bev, goal):
        # TODO: build input dict matching exported ONNX I/O names
        gvec = np.array([[goal.dx, goal.dy, goal.distance, goal.heading_error]], np.float32)
        bin_ = bev[None].astype(np.float32)
        try:
            names = [i.name for i in self.session.get_inputs()]
            out = self.session.run(None, {names[0]: bin_, names[1]: gvec})[0][0]
            return float(out[0]), float(out[1]), float(out[2]), float(out[3] if len(out) > 3 else 0.0)
        except Exception as e:
            self.get_logger().warn(f'onnx infer failed ({e}); zero cmd')
            return 0.0, 0.0, 0.0, 1.0

    def tick(self):
        t0 = time.time()
        bev = self._bev_array()
        valid = bev is not None and self.enabled
        if not valid:
            vx = vy = wz = 0.0
            unc = 1.0
        elif self.runtime_type == 'onnx' and self.session is not None:
            vx, vy, wz, unc = self._infer_onnx(bev, self.goal)
        else:
            vx, vy, wz, unc = self._infer_dummy(bev, self.goal)

        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = vx, vy, wz
        self.pub_cmd.publish(cmd)

        dbg = PolicyDebug()
        dbg.header.stamp = self.get_clock().now().to_msg()
        dbg.vx_raw, dbg.vy_raw, dbg.wz_raw = vx, vy, wz
        dbg.uncertainty = unc
        dbg.inference_time_ms = (time.time() - t0) * 1000.0
        dbg.runtime_type = self.runtime_type
        dbg.policy_valid = bool(valid)
        self.pub_dbg.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyRuntime()
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
