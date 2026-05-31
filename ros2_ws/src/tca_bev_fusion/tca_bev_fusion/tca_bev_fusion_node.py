#!/usr/bin/env python3
"""TCA-BEV: Time- and Calibration-Aware Bird's-Eye-View fusion.

Builds a body-centric (base_link / anchor) BEV tensor from a 360 deg LiDAR
point cloud and a forward depth image, while explicitly encoding time
confidence, calibration confidence and pose-anchor quality so that the
downstream policy is aware of *how trustworthy* each part of the BEV is.

Stage-1 implementation is in Python for portability; the perf-sensitive
projection loop is vectorized with numpy. A C++/rclcpp port is planned for
the Jetson edge build (see docs/tca_bev_method.md, TODO: cpp_port).

Channels (configurable, first 8 implemented, rest are placeholders):
  0 lidar_occupancy        4 depth_occupancy        8  time_confidence_map
  1 lidar_height_max       5 depth_near_obstacle     9  calibration_confidence_map
  2 lidar_density          6 unknown_mask           10  anchor_quality_map
  3 lidar_free_space       7 dynamic_decay          11  goal_direction_map_x
                                                    12  goal_direction_map_y

Fusion modes:
  lidar_only | depth_only | naive | tca   (default: tca)
"""
import math
import struct
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from nav_msgs.msg import Odometry
from e2e_nav_msgs.msg import BevTensor, AnchorStatus, SensorStatus, GoalVector

CHANNEL_NAMES = [
    'lidar_occupancy', 'lidar_height_max', 'lidar_density', 'lidar_free_space',
    'depth_occupancy', 'depth_near_obstacle', 'unknown_mask', 'dynamic_decay',
    'time_confidence_map', 'calibration_confidence_map', 'anchor_quality_map',
    'goal_direction_map_x', 'goal_direction_map_y',
]


def read_xyz(cloud: PointCloud2):
    """Vectorized parse of xyz from a PointCloud2 (assumes float32 x,y,z first)."""
    if cloud.width * cloud.height == 0:
        return np.zeros((0, 3), np.float32)
    n = cloud.width * cloud.height
    step = cloud.point_step
    raw = np.frombuffer(bytes(cloud.data), dtype=np.uint8)
    raw = raw[: n * step].reshape(n, step)
    # offsets of x,y,z
    off = {f.name: f.offset for f in cloud.fields}
    ox, oy, oz = off.get('x', 0), off.get('y', 4), off.get('z', 8)
    def col(o):
        return raw[:, o:o + 4].copy().view(np.float32).reshape(-1)
    xyz = np.stack([col(ox), col(oy), col(oz)], axis=1)
    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask]


class TcaBevFusion(Node):
    def __init__(self):
        super().__init__('tca_bev_fusion')
        self.declare_parameter('size_x', 5.0)
        self.declare_parameter('size_y', 5.0)
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('update_rate', 10.0)
        self.declare_parameter('frame', 'base_link')
        self.declare_parameter('fusion_mode', 'tca')          # lidar_only|depth_only|naive|tca
        self.declare_parameter('num_channels', 13)
        self.declare_parameter('z_min', -0.3)
        self.declare_parameter('z_max', 1.2)
        self.declare_parameter('depth_max', 5.0)
        self.declare_parameter('depth_near_thresh', 1.0)
        self.declare_parameter('camera_xyz', [0.20, 0.0, 0.20])  # base->camera
        self.declare_parameter('camera_yaw', 0.0)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('dynamic_decay', 0.85)

        gp = self.get_parameter
        self.size_x = float(gp('size_x').value)
        self.size_y = float(gp('size_y').value)
        self.res = float(gp('resolution').value)
        self.frame = gp('frame').value
        self.fusion_mode = gp('fusion_mode').value
        self.C = int(gp('num_channels').value)
        self.z_min = float(gp('z_min').value)
        self.z_max = float(gp('z_max').value)
        self.depth_max = float(gp('depth_max').value)
        self.depth_near = float(gp('depth_near_thresh').value)
        self.cam_xyz = list(gp('camera_xyz').value)
        self.cam_yaw = float(gp('camera_yaw').value)
        self.publish_debug = bool(gp('publish_debug_image').value)
        self.decay = float(gp('dynamic_decay').value)

        # BEV grid is body-centric: origin at robot, +x forward.
        self.H = int(round(self.size_x / self.res))
        self.W = int(round(self.size_y / self.res))
        self.origin_x = -self.size_x / 2.0
        self.origin_y = -self.size_y / 2.0

        self.get_logger().info(
            f"[tca_bev] mode={self.fusion_mode} grid={self.H}x{self.W} "
            f"res={self.res} channels={self.C}")

        # latest inputs
        self.lidar = None
        self.depth = None
        self.cinfo = None
        self.goal = None
        self.anchor_quality = 1.0
        self.anchor_valid = True
        self.time_conf = 1.0
        self.calib_conf = 0.5
        self.depth_inflation = 0.3
        self.prev_occ = np.zeros((self.H, self.W), np.float32)

        self.create_subscription(PointCloud2, '/aligned/lidar_points', self.cb_lidar, 5)
        self.create_subscription(Image, '/aligned/depth_image', self.cb_depth, 5)
        self.create_subscription(CameraInfo, '/aligned/camera_info', self.cb_cinfo, 5)
        self.create_subscription(GoalVector, '/goal/vector', self.cb_goal, 5)
        self.create_subscription(AnchorStatus, '/anchor/status', self.cb_anchor, 5)
        self.create_subscription(SensorStatus, '/diagnostics/time_align_status', self.cb_time, 5)
        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(SensorStatus, '/calibration/status', self.cb_calib, qos)

        self.pub_tensor = self.create_publisher(BevTensor, '/bev/tensor', 5)
        self.pub_status = self.create_publisher(SensorStatus, '/bev/status', 5)
        if self.publish_debug:
            self.pub_dbg = self.create_publisher(Image, '/bev/image_debug', 1)

        self.create_timer(1.0 / float(gp('update_rate').value), self.tick)

    # --- callbacks ---
    def cb_lidar(self, msg): self.lidar = msg
    def cb_depth(self, msg): self.depth = msg
    def cb_cinfo(self, msg): self.cinfo = msg
    def cb_goal(self, msg): self.goal = msg

    def cb_anchor(self, msg):
        self.anchor_quality = float(msg.anchor_quality)
        self.anchor_valid = bool(msg.is_valid)

    def cb_time(self, msg):
        self.time_conf = float(msg.valid_ratio)

    def cb_calib(self, msg):
        self.calib_conf = float(msg.valid_ratio)
        self.depth_inflation = float(msg.last_time_diff_ms) / 1000.0

    # --- BEV index helpers ---
    def world_to_cell(self, x, y):
        # +x forward -> row index increases forward (i=H-1 top); col from y
        ci = ((x - self.origin_x) / self.res).astype(np.int32)
        cj = ((y - self.origin_y) / self.res).astype(np.int32)
        valid = (ci >= 0) & (ci < self.H) & (cj >= 0) & (cj < self.W)
        return ci, cj, valid

    def build_lidar_channels(self):
        occ = np.zeros((self.H, self.W), np.float32)
        hmax = np.zeros((self.H, self.W), np.float32)
        dens = np.zeros((self.H, self.W), np.float32)
        free = np.zeros((self.H, self.W), np.float32)
        if self.lidar is None:
            return occ, hmax, dens, free
        xyz = read_xyz(self.lidar)
        if xyz.shape[0] == 0:
            return occ, hmax, dens, free
        zmask = (xyz[:, 2] >= self.z_min) & (xyz[:, 2] <= self.z_max)
        xyz = xyz[zmask]
        ci, cj, valid = self.world_to_cell(xyz[:, 0], xyz[:, 1])
        ci, cj, zz = ci[valid], cj[valid], xyz[valid, 2]
        np.add.at(dens, (ci, cj), 1.0)
        np.maximum.at(hmax, (ci, cj), zz)
        occ[dens > 0] = 1.0
        if dens.max() > 0:
            dens = dens / dens.max()
        # crude free-space: cells along ray to each hit (approx by marking the
        # straight segment from center to hit as free where not occupied)
        free = self._ray_free(ci, cj)
        free[occ > 0] = 0.0
        return occ, hmax, dens, free

    def _ray_free(self, ci, cj):
        free = np.zeros((self.H, self.W), np.float32)
        cx = int((0.0 - self.origin_x) / self.res)
        cy = int((0.0 - self.origin_y) / self.res)
        # subsample hits for speed
        idx = np.arange(0, len(ci), max(1, len(ci) // 360))
        for k in idx:
            r0, c0, r1, c1 = cx, cy, int(ci[k]), int(cj[k])
            steps = max(abs(r1 - r0), abs(c1 - c0), 1)
            rr = np.linspace(r0, r1, steps).astype(np.int32)
            cc = np.linspace(c0, c1, steps).astype(np.int32)
            m = (rr >= 0) & (rr < self.H) & (cc >= 0) & (cc < self.W)
            free[rr[m], cc[m]] = 1.0
        return free

    def build_depth_channels(self):
        docc = np.zeros((self.H, self.W), np.float32)
        dnear = np.zeros((self.H, self.W), np.float32)
        if self.depth is None or self.cinfo is None:
            return docc, dnear
        try:
            d = np.frombuffer(bytes(self.depth.data), dtype=np.float32).reshape(
                self.depth.height, self.depth.width).copy()
        except Exception:
            return docc, dnear
        fx = self.cinfo.k[0] or 1.0
        cx = self.cinfo.k[2]
        cols = np.arange(self.depth.width)
        # per-column nearest valid depth (mock depth is flat per column)
        zc = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=0)
        valid = np.isfinite(zc) & (zc > 0.05) & (zc < self.depth_max)
        zc = zc[valid]
        cols = cols[valid]
        if zc.size == 0:
            return docc, dnear
        # ray angle from pinhole; point in camera optical -> camera body (x fwd)
        ang = np.arctan2((cols - cx), fx)        # left/right angle
        # camera body frame: x forward = zc*cos(ang), y left = -zc*sin(ang)
        xb = zc * np.cos(ang)
        yb = -zc * np.sin(ang)
        # transform camera->base (yaw + translation)
        cyaw, syaw = math.cos(self.cam_yaw), math.sin(self.cam_yaw)
        xw = self.cam_xyz[0] + cyaw * xb - syaw * yb
        yw = self.cam_xyz[1] + syaw * xb + cyaw * yb
        ci, cj, vmask = self.world_to_cell(xw, yw)
        ci, cj, zc = ci[vmask], cj[vmask], zc[vmask]
        docc[ci, cj] = 1.0
        near = zc < self.depth_near
        dnear[ci[near], cj[near]] = 1.0
        # conservative inflation when calibration confidence is low
        infl_cells = int(round(self.depth_inflation / self.res))
        if infl_cells > 0:
            docc = self._inflate(docc, infl_cells)
            dnear = self._inflate(dnear, infl_cells)
        return docc, dnear

    @staticmethod
    def _inflate(grid, radius):
        out = grid.copy()
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc > radius * radius:
                    continue
                out = np.maximum(out, np.roll(np.roll(grid, dr, 0), dc, 1))
        return out

    def fuse_occupancy(self, lidar_occ, depth_occ):
        lc = self.lidar_modality_conf()
        dc = self.depth_modality_conf()
        if self.fusion_mode == 'lidar_only':
            return lidar_occ, lc, 0.0
        if self.fusion_mode == 'depth_only':
            return depth_occ, 0.0, dc
        if self.fusion_mode == 'naive':
            return np.maximum(lidar_occ, depth_occ), lc, dc
        # tca: probabilistic OR with per-modality confidence
        p = 1.0 - (1.0 - lc * lidar_occ) * (1.0 - dc * depth_occ)
        return p, lc, dc

    def lidar_modality_conf(self):
        # lidar extrinsic is fairly trusted; scale by time conf and anchor quality
        return float(np.clip(self.time_conf * (0.5 + 0.5 * self.anchor_quality), 0.0, 1.0))

    def depth_modality_conf(self):
        return float(np.clip(self.time_conf * self.calib_conf *
                             (0.5 + 0.5 * self.anchor_quality), 0.0, 1.0))

    def tick(self):
        t0 = time.time()
        l_occ, l_hmax, l_dens, l_free = self.build_lidar_channels()
        d_occ, d_near = self.build_depth_channels()
        fused, lc, dc = self.fuse_occupancy(l_occ, d_occ)

        # unknown mask: neither observed as free nor occupied
        observed = np.maximum(np.maximum(l_free, l_occ), d_occ)
        unknown = (observed < 1e-3).astype(np.float32)

        # dynamic decay (temporal): keep memory of past occupancy
        dyn = np.maximum(fused, self.prev_occ * self.decay)
        self.prev_occ = dyn

        chans = np.zeros((self.C, self.H, self.W), np.float32)
        layers = [fused, l_hmax, l_dens, l_free, d_occ, d_near, unknown, dyn,
                  np.full((self.H, self.W), self.time_conf, np.float32),
                  np.full((self.H, self.W), self.calib_conf, np.float32),
                  np.full((self.H, self.W), self.anchor_quality, np.float32)]
        # goal direction maps
        if self.goal is not None and self.goal.has_goal and self.C > 12:
            gx = np.full((self.H, self.W), self.goal.dx, np.float32)
            gy = np.full((self.H, self.W), self.goal.dy, np.float32)
            norm = max(1e-3, math.hypot(self.goal.dx, self.goal.dy))
            layers += [gx / norm, gy / norm]
        for i in range(min(self.C, len(layers))):
            chans[i] = layers[i]

        global_conf = float(np.clip(self.time_conf * (0.5 + 0.5 * self.calib_conf) *
                                    (0.5 + 0.5 * self.anchor_quality), 0.0, 1.0))

        self._publish_tensor(chans, lc, dc, global_conf)
        latency = (time.time() - t0) * 1000.0
        self._publish_status(latency)
        if self.publish_debug:
            self._publish_debug(fused, unknown, d_near)

    def _publish_tensor(self, chans, lc, dc, gconf):
        msg = BevTensor()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame
        msg.channels = self.C
        msg.height = self.H
        msg.width = self.W
        msg.resolution = self.res
        msg.origin_x = self.origin_x
        msg.origin_y = self.origin_y
        msg.data = chans.reshape(-1).astype(np.float32).tolist()
        msg.global_confidence = gconf
        msg.lidar_confidence = float(lc)
        msg.depth_confidence = float(dc)
        msg.calibration_confidence = float(self.calib_conf)
        msg.time_confidence = float(self.time_conf)
        msg.anchor_quality = float(self.anchor_quality)
        self.pub_tensor.publish(msg)

    def _publish_status(self, latency_ms):
        st = SensorStatus()
        st.header.stamp = self.get_clock().now().to_msg()
        st.sensor_name = 'tca_bev'
        st.alive = True
        st.last_time_diff_ms = float(latency_ms)
        st.frame_id = self.frame
        st.message = f"mode={self.fusion_mode} latency={latency_ms:.1f}ms"
        self.pub_status.publish(st)

    def _publish_debug(self, occ, unknown, dnear):
        # RGB: occupancy=red, unknown=gray, depth-near=blue overlay
        img = np.zeros((self.H, self.W, 3), np.uint8)
        img[..., 0] = (occ * 255).astype(np.uint8)
        g = (unknown * 90).astype(np.uint8)
        img[..., 1] = np.maximum(img[..., 1], g)
        img[..., 2] = np.maximum(img[..., 2], (dnear * 255).astype(np.uint8))
        img[..., 1] = np.maximum(img[..., 1], (dnear * 120).astype(np.uint8))
        # flip so forward (+x) is up
        img = np.flipud(img)
        m = Image()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame
        m.height, m.width = self.H, self.W
        m.encoding = 'rgb8'
        m.is_bigendian = 0
        m.step = self.W * 3
        m.data = img.tobytes()
        self.pub_dbg.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = TcaBevFusion()
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
