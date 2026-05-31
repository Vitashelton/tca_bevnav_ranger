#!/usr/bin/env python3
"""BEV / policy debug visualization for RViz and PC.

Publishes:
  /bev/markers       (MarkerArray): goal direction, policy & safe velocity arrows
  /bev/anchor_quality (visualization text)
Consumes the BEV tensor, goal vector, raw/safe cmd and anchor status.
"""
import math
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Twist, Point
from e2e_nav_msgs.msg import GoalVector, AnchorStatus, BevTensor


class BevVisualization(Node):
    def __init__(self):
        super().__init__('bev_visualization')
        self.declare_parameter('frame', 'base_link')
        self.frame = self.get_parameter('frame').value
        self.goal = None
        self.raw = (0.0, 0.0, 0.0)
        self.safe = (0.0, 0.0, 0.0)
        self.anchor_q = 1.0
        self.get_logger().info(f"[bev_viz] frame={self.frame}")

        self.create_subscription(GoalVector, '/goal/vector', self._g, 5)
        self.create_subscription(Twist, '/cmd_vel_raw', self._r, 5)
        self.create_subscription(Twist, '/cmd_vel_safe', self._s, 5)
        self.create_subscription(AnchorStatus, '/anchor/status', self._a, 5)
        self.pub = self.create_publisher(MarkerArray, '/bev/markers', 5)
        self.create_timer(0.1, self.tick)

    def _g(self, m): self.goal = m
    def _r(self, m): self.raw = (m.linear.x, m.linear.y, m.angular.z)
    def _s(self, m): self.safe = (m.linear.x, m.linear.y, m.angular.z)
    def _a(self, m): self.anchor_q = float(m.anchor_quality)

    def _arrow(self, mid, vx, vy, rgb, scale=1.0):
        mk = Marker()
        mk.header.frame_id = self.frame
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns = 'bev_viz'
        mk.id = mid
        mk.type = Marker.ARROW
        mk.action = Marker.ADD
        p0 = Point(x=0.0, y=0.0, z=0.1)
        p1 = Point(x=float(vx * scale), y=float(vy * scale), z=0.1)
        mk.points = [p0, p1]
        mk.scale.x = 0.05; mk.scale.y = 0.1; mk.scale.z = 0.1
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = (*rgb, 1.0)
        return mk

    def tick(self):
        arr = MarkerArray()
        if self.goal is not None and self.goal.has_goal:
            arr.markers.append(self._arrow(
                0, math.cos(self.goal.heading_error), math.sin(self.goal.heading_error),
                (0.1, 0.8, 0.1)))
        arr.markers.append(self._arrow(1, self.raw[0], self.raw[1], (0.9, 0.6, 0.1)))
        arr.markers.append(self._arrow(2, self.safe[0], self.safe[1], (0.1, 0.4, 0.9)))
        txt = Marker()
        txt.header.frame_id = self.frame
        txt.header.stamp = self.get_clock().now().to_msg()
        txt.ns = 'bev_viz'; txt.id = 3; txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.z = 0.6
        txt.scale.z = 0.15
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
        txt.text = f"anchor_q={self.anchor_q:.2f}"
        arr.markers.append(txt)
        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    n = BevVisualization()
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
