#!/usr/bin/env python3
"""Local goal manager.

Converts a goal (from RViz 2D Nav Goal, a script, or a fixed/waypoint
parameter) into a body-frame goal vector for the end-to-end policy. This
is deliberately NOT a global planner: it only produces (dx, dy, distance,
heading_error). If there is no goal, has_goal=False so the policy stops.
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from e2e_nav_msgs.msg import GoalVector


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class GoalManager(Node):
    def __init__(self):
        super().__init__('goal_manager')
        self.declare_parameter('goal_topic', '/goal/raw')
        self.declare_parameter('odom_topic', '/anchor/odom')
        self.declare_parameter('goal_reached_dist', 0.25)
        self.declare_parameter('waypoints', [])     # flat [x0,y0,x1,y1,...]
        self.declare_parameter('use_waypoints', False)
        self.declare_parameter('publish_rate', 10.0)

        gp = self.get_parameter
        self.reached = float(gp('goal_reached_dist').value)
        self.use_wp = bool(gp('use_waypoints').value)
        wp = list(gp('waypoints').value)
        self.waypoints = [(wp[i], wp[i + 1]) for i in range(0, len(wp) - 1, 2)]
        self.wp_idx = 0

        self.goal_world = None          # (x, y) in odom/anchor frame
        self.robot = (0.0, 0.0, 0.0)    # x, y, yaw

        self.get_logger().info(
            f"[goal_manager] use_waypoints={self.use_wp} n_wp={len(self.waypoints)} "
            f"reached_dist={self.reached}")

        self.create_subscription(PoseStamped, gp('goal_topic').value, self.cb_goal, 1)
        self.create_subscription(Odometry, gp('odom_topic').value, self.cb_odom, 10)
        self.pub_vec = self.create_publisher(GoalVector, '/goal/vector', 5)
        self.pub_local = self.create_publisher(PoseStamped, '/goal/local', 1)
        self.create_timer(1.0 / float(gp('publish_rate').value), self.tick)

        if self.use_wp and self.waypoints:
            self.goal_world = self.waypoints[0]

    def cb_goal(self, msg):
        self.goal_world = (msg.pose.position.x, msg.pose.position.y)
        self.use_wp = False

    def cb_odom(self, msg):
        self.robot = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                      yaw_from_quat(msg.pose.pose.orientation))

    def tick(self):
        gv = GoalVector()
        gv.header.stamp = self.get_clock().now().to_msg()
        gv.header.frame_id = 'base_link'
        if self.goal_world is None:
            gv.has_goal = False
            self.pub_vec.publish(gv)
            return
        rx, ry, ryaw = self.robot
        gx, gy = self.goal_world
        # world delta -> body frame
        dwx, dwy = gx - rx, gy - ry
        c, s = math.cos(-ryaw), math.sin(-ryaw)
        dx = c * dwx - s * dwy
        dy = s * dwx + c * dwy
        dist = math.hypot(dx, dy)
        heading = math.atan2(dy, dx)

        if dist < self.reached:
            if self.use_wp and self.wp_idx + 1 < len(self.waypoints):
                self.wp_idx += 1
                self.goal_world = self.waypoints[self.wp_idx]
            else:
                gv.has_goal = False
                self.pub_vec.publish(gv)
                return

        gv.dx, gv.dy = float(dx), float(dy)
        gv.distance = float(dist)
        gv.heading_error = float(heading)
        gv.has_goal = True
        self.pub_vec.publish(gv)

        lp = PoseStamped()
        lp.header = gv.header
        lp.pose.position.x = dx
        lp.pose.position.y = dy
        lp.pose.orientation.w = 1.0
        self.pub_local.publish(lp)


def main(args=None):
    rclpy.init(args=args)
    node = GoalManager()
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
