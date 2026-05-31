#!/usr/bin/env python3
"""Mock sensor generator for hardware-free closed-loop testing.

Generates a simple parametric indoor scene (walls, a door, a narrow
corridor, static obstacles and one moving "pedestrian") and publishes:
  - /sensors/lidar_points   (sensor_msgs/PointCloud2)
  - /sensors/depth_image    (sensor_msgs/Image, 32FC1, meters)
  - /sensors/camera_info    (sensor_msgs/CameraInfo)
  - /mock/odom              (nav_msgs/Odometry)
  - /goal/raw               (geometry_msgs/PoseStamped)

All shapes are intentionally simple. This is NOT a physics simulator; it
only needs to exercise the downstream BEV / policy / safety pipeline.
"""
import math
import struct

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


def make_pointcloud2(header, points):
    """points: Nx3 float32 array -> PointCloud2 (xyz)."""
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = True
    msg.data = points.astype(np.float32).tobytes()
    return msg


class MockSensorsNode(Node):
    def __init__(self):
        super().__init__('mock_sensors_node')

        # ---- parameters ----
        self.declare_parameter('scenario', 'corridor_door')
        self.declare_parameter('lidar_rate', 10.0)
        self.declare_parameter('depth_rate', 15.0)
        self.declare_parameter('odom_rate', 50.0)
        self.declare_parameter('lidar_frame', 'lidar_link')
        self.declare_parameter('depth_frame', 'camera_depth_optical_frame')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('depth_width', 160)
        self.declare_parameter('depth_height', 120)
        self.declare_parameter('lidar_range', 6.0)
        self.declare_parameter('lidar_beams', 720)
        self.declare_parameter('publish_goal', True)
        self.declare_parameter('goal_x', 4.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('dynamic_obstacle', True)

        gp = self.get_parameter
        self.scenario = gp('scenario').value
        self.lidar_frame = gp('lidar_frame').value
        self.depth_frame = gp('depth_frame').value
        self.odom_frame = gp('odom_frame').value
        self.base_frame = gp('base_frame').value
        self.dw = int(gp('depth_width').value)
        self.dh = int(gp('depth_height').value)
        self.lidar_range = float(gp('lidar_range').value)
        self.lidar_beams = int(gp('lidar_beams').value)
        self.publish_goal = bool(gp('publish_goal').value)
        self.goal_x = float(gp('goal_x').value)
        self.goal_y = float(gp('goal_y').value)
        self.dynamic = bool(gp('dynamic_obstacle').value)

        self.get_logger().info(
            f"[mock_sensors] scenario={self.scenario} lidar={self.lidar_frame} "
            f"depth={self.dw}x{self.dh} goal=({self.goal_x},{self.goal_y})")

        # ---- publishers ----
        self.pub_lidar = self.create_publisher(PointCloud2, '/sensors/lidar_points', 5)
        self.pub_depth = self.create_publisher(Image, '/sensors/depth_image', 5)
        self.pub_cinfo = self.create_publisher(CameraInfo, '/sensors/camera_info', 5)
        self.pub_odom = self.create_publisher(Odometry, '/mock/odom', 10)
        self.pub_goal = self.create_publisher(PoseStamped, '/goal/raw', 1)

        # ---- timers ----
        self.create_timer(1.0 / float(gp('lidar_rate').value), self.tick_lidar)
        self.create_timer(1.0 / float(gp('depth_rate').value), self.tick_depth)
        self.create_timer(1.0 / float(gp('odom_rate').value), self.tick_odom)
        if self.publish_goal:
            self.create_timer(1.0, self.tick_goal)

        self.t0 = self.get_clock().now()
        self.cinfo = self._build_camera_info()

    # ---------------------------------------------------------------
    def _elapsed(self):
        return (self.get_clock().now() - self.t0).nanoseconds * 1e-9

    def _now_header(self, frame):
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = frame
        return h

    # ---------------------------------------------------------------
    # Scene geometry: list of obstacle segments / circles in base_link XY.
    def _scene_obstacles(self, t):
        """Return (segments, circles).
        segments: list of (x1,y1,x2,y2) walls.
        circles : list of (cx,cy,r)."""
        segments = []
        circles = []
        if self.scenario in ('corridor_door', 'default'):
            # corridor walls along +x, width ~1.4m, with a door gap at x in [2.0,2.8]
            wy = 0.7
            # left wall (y=+wy) with door gap
            segments += [(0.5, wy, 2.0, wy), (2.8, wy, 5.5, wy)]
            # right wall (y=-wy) solid
            segments += [(0.5, -wy, 5.5, -wy)]
            # door frame posts
            segments += [(2.0, wy, 2.0, wy - 0.15), (2.8, wy, 2.8, wy - 0.15)]
            # a static box obstacle
            circles += [(1.3, -0.25, 0.18)]
        elif self.scenario == 'open_room':
            segments += [(-1, 3, 6, 3), (-1, -3, 6, -3), (6, -3, 6, 3)]
            circles += [(2.0, 0.6, 0.25), (3.0, -0.8, 0.25)]
        elif self.scenario == 'narrow_corridor':
            wy = 0.45
            segments += [(0.5, wy, 6.0, wy), (0.5, -wy, 6.0, -wy)]
        # moving pedestrian
        if self.dynamic:
            px = 3.0
            py = 0.6 * math.sin(0.5 * t)
            circles += [(px, py, 0.22)]
        return segments, circles

    @staticmethod
    def _ray_hit(angle, segments, circles, max_r):
        """Cast a ray from origin at given angle; return distance to nearest hit."""
        dx, dy = math.cos(angle), math.sin(angle)
        best = max_r
        # segments
        for (x1, y1, x2, y2) in segments:
            ex, ey = x2 - x1, y2 - y1
            denom = dx * ey - dy * ex
            if abs(denom) < 1e-9:
                continue
            t = ((x1 * ey - y1 * ex)) / denom          # along ray
            s = ((x1 * dy - y1 * dx)) / denom           # along segment
            if t > 0.02 and 0.0 <= s <= 1.0 and t < best:
                best = t
        # circles
        for (cx, cy, r) in circles:
            b = -(dx * cx + dy * cy)
            c = cx * cx + cy * cy - r * r
            disc = b * b - c
            if disc >= 0:
                t = -b - math.sqrt(disc)
                if 0.02 < t < best:
                    best = t
        return best

    # ---------------------------------------------------------------
    def tick_lidar(self):
        t = self._elapsed()
        segs, circs = self._scene_obstacles(t)
        pts = []
        for i in range(self.lidar_beams):
            ang = -math.pi + 2 * math.pi * i / self.lidar_beams
            r = self._ray_hit(ang, segs, circs, self.lidar_range)
            if r < self.lidar_range - 1e-3:
                x = r * math.cos(ang)
                y = r * math.sin(ang)
                # a couple of z layers to look 3D
                for z in (0.0, 0.2, -0.1):
                    pts.append((x, y, z))
        arr = np.array(pts, dtype=np.float32) if pts else np.zeros((1, 3), np.float32)
        self.pub_lidar.publish(make_pointcloud2(self._now_header(self.lidar_frame), arr))

    def tick_depth(self):
        t = self._elapsed()
        segs, circs = self._scene_obstacles(t)
        # forward-facing depth: horizontal FOV ~86deg (D435i), simple per-column raycast
        fov = math.radians(86.0)
        depth = np.full((self.dh, self.dw), np.nan, dtype=np.float32)
        for u in range(self.dw):
            ang = (u / (self.dw - 1) - 0.5) * fov   # angle relative to +x
            r = self._ray_hit(ang, segs, circs, 5.0)
            if r < 5.0 - 1e-3:
                depth[:, u] = r   # flat column (no vertical structure in mock)
        h = self._now_header(self.depth_frame)
        msg = Image()
        msg.header = h
        msg.height = self.dh
        msg.width = self.dw
        msg.encoding = '32FC1'
        msg.is_bigendian = 0
        msg.step = self.dw * 4
        msg.data = depth.tobytes()
        self.pub_depth.publish(msg)
        ci = self.cinfo
        ci.header = h
        self.pub_cinfo.publish(ci)

    def tick_odom(self):
        # robot stays at origin in mock (BEV is body-centric); odom is identity
        # with a tiny drift to exercise anchor quality estimation.
        t = self._elapsed()
        msg = Odometry()
        msg.header = self._now_header(self.odom_frame)
        msg.child_frame_id = self.base_frame
        msg.pose.pose.position.x = 0.0
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.orientation.w = 1.0
        # small noise-free covariance
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.01
        cov[35] = 0.02
        msg.pose.covariance = cov
        self.pub_odom.publish(msg)

    def tick_goal(self):
        msg = PoseStamped()
        msg.header = self._now_header(self.odom_frame)
        msg.pose.position.x = self.goal_x
        msg.pose.position.y = self.goal_y
        msg.pose.orientation.w = 1.0
        self.pub_goal.publish(msg)

    def _build_camera_info(self):
        ci = CameraInfo()
        ci.width = self.dw
        ci.height = self.dh
        fx = self.dw / (2.0 * math.tan(math.radians(86.0) / 2.0))
        fy = fx
        cx, cy = self.dw / 2.0, self.dh / 2.0
        ci.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        ci.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        ci.distortion_model = 'plumb_bob'
        ci.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        return ci


def main(args=None):
    rclpy.init(args=args)
    node = MockSensorsNode()
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
