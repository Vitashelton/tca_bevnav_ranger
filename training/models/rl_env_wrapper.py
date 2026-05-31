#!/usr/bin/env python3
"""Gym-style environment wrapper for PPO fine-tuning of the BEV policy.

This is a skeleton / interface definition. Two concrete backends are intended:

  1. ROS2 closed-loop env  -- steps the real mock pipeline (mock_sensors ->
     time_align -> tca_bev_fusion) and reads /bev/tensor, applies the action
     via /cmd_vel_raw, and reads back odom for reward. Useful for sim-in-the-
     loop using the existing nodes. (TODO: wire rclpy bridge.)

  2. Lightweight 2D kinematic env -- a fast pure-numpy simulator that re-uses
     the same scene primitives as mock_sensors (walls, door gap, moving
     pedestrian) and renders an approximate BEV. This is what train_ppo.py
     uses by default because it does not require a running ROS2 graph.

Only the lightweight env is implemented here; the ROS2 backend is stubbed with
clear TODOs so it does not fabricate behavior.

Observation: dict(bev=(C,H,W), goal=(4,), vel=(3,))
Action:      np.array([vx, vy, wz])
Reward terms (see docs/training_strategy.md):
  progress_to_goal, heading_alignment, collision_penalty,
  near_obstacle_penalty, door_passing_reward,
  narrow_corridor_centering_reward, smooth_velocity_reward,
  time_penalty, success_reward
"""
import numpy as np

BEV_CHANNELS = 13


class RewardConfig:
    progress = 1.0
    heading = 0.1
    collision = -10.0
    near_obstacle = -0.5
    door_passing = 2.0
    corridor_centering = 0.3
    smooth = -0.05
    time = -0.01
    success = 20.0


class KinematicBevEnv:
    """Fast 2D omni-base simulator producing an approximate TCA-BEV.

    The dynamics are a simple integrator on (x, y, theta) with omni velocity
    (vx, vy in body frame, wz). Obstacles are line segments (walls) plus a
    moving circular pedestrian, matching the mock_sensors scene families so a
    policy trained here transfers to the mock ROS2 pipeline.
    """

    def __init__(self, bev_size=100, resolution=0.05, dt=0.1,
                 max_steps=300, reward_cfg=None, seed=0):
        self.H = self.W = bev_size
        self.res = resolution
        self.dt = dt
        self.max_steps = max_steps
        self.rcfg = reward_cfg or RewardConfig()
        self.rng = np.random.default_rng(seed)
        self.vmax = np.array([0.6, 0.4, 1.0], dtype=np.float32)
        self._reset_state()

    # ---- scene -----------------------------------------------------------
    def _reset_state(self):
        # robot pose in world frame
        self.pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.goal = np.array([3.0, self.rng.uniform(-0.5, 0.5)], dtype=np.float32)
        # walls forming a corridor with a door gap
        gap = self.rng.uniform(0.7, 1.1)
        cy = self.rng.uniform(-0.3, 0.3)
        self.walls = [
            ((1.5, cy + gap / 2), (1.5, 2.5)),
            ((1.5, cy - gap / 2), (1.5, -2.5)),
        ]
        self.ped = np.array([2.2, self.rng.uniform(-0.5, 0.5)], dtype=np.float32)
        self.ped_v = np.array([0.0, self.rng.choice([-1, 1]) * 0.2], dtype=np.float32)
        self.steps = 0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.prev_dist = np.linalg.norm(self.goal - self.pose[:2])

    def reset(self):
        self._reset_state()
        return self._observe()

    # ---- observation -----------------------------------------------------
    def _world_to_bev(self, pts):
        """pts: (N,2) world -> body -> bev indices. Returns valid (row,col)."""
        c, s = np.cos(-self.pose[2]), np.sin(-self.pose[2])
        rel = pts - self.pose[:2]
        bx = rel[:, 0] * c - rel[:, 1] * s
        by = rel[:, 0] * s + rel[:, 1] * c
        # body x forward -> decreasing row (row 0 farthest ahead)
        col = (self.W / 2 + by / self.res).astype(int)
        row = (self.H / 2 - bx / self.res).astype(int)
        m = (row >= 0) & (row < self.H) & (col >= 0) & (col < self.W)
        return row[m], col[m]

    def _sample_walls(self, n=200):
        pts = []
        for (a, b) in self.walls:
            t = np.linspace(0, 1, n)
            xs = a[0] + (b[0] - a[0]) * t
            ys = a[1] + (b[1] - a[1]) * t
            pts.append(np.stack([xs, ys], 1))
        return np.concatenate(pts, 0)

    def _observe(self):
        bev = np.zeros((BEV_CHANNELS, self.H, self.W), dtype=np.float32)
        wall_pts = self._sample_walls()
        r, c = self._world_to_bev(wall_pts)
        bev[0, r, c] = 1.0                      # lidar_occupancy
        # pedestrian as a small blob in depth + occupancy
        ped_pts = self.ped[None, :] + self.rng.normal(0, 0.05, (30, 2))
        pr, pc = self._world_to_bev(ped_pts)
        bev[0, pr, pc] = 1.0
        bev[4, pr, pc] = 1.0                    # depth_occupancy
        # confidence maps (constant here; real pipeline fills these)
        bev[8].fill(0.9)                        # time_confidence_map
        bev[9].fill(0.4)                        # calibration_confidence_map
        bev[10].fill(0.8)                       # anchor_quality_map
        goal_body = self._goal_body()
        dist = np.linalg.norm(goal_body)
        heading = np.arctan2(goal_body[1], goal_body[0])
        goal_vec = np.array([goal_body[0], goal_body[1], dist, heading], np.float32)
        return {'bev': bev, 'goal': goal_vec, 'vel': self.vel.copy()}

    def _goal_body(self):
        c, s = np.cos(-self.pose[2]), np.sin(-self.pose[2])
        rel = self.goal - self.pose[:2]
        return np.array([rel[0] * c - rel[1] * s, rel[0] * s + rel[1] * c], np.float32)

    # ---- step ------------------------------------------------------------
    def step(self, action):
        action = np.clip(action, -self.vmax, self.vmax).astype(np.float32)
        # integrate body-frame omni velocity into world frame
        th = self.pose[2]
        vx, vy, wz = action
        self.pose[0] += (vx * np.cos(th) - vy * np.sin(th)) * self.dt
        self.pose[1] += (vx * np.sin(th) + vy * np.cos(th)) * self.dt
        self.pose[2] += wz * self.dt
        self.vel = action
        self.ped += self.ped_v * self.dt
        if abs(self.ped[1]) > 0.6:
            self.ped_v[1] *= -1
        self.steps += 1

        dist = np.linalg.norm(self.goal - self.pose[:2])
        reward, done, info = self._reward(action, dist)
        self.prev_dist = dist
        self.prev_action = action
        return self._observe(), reward, done, info

    def _min_obstacle_dist(self):
        wall_pts = self._sample_walls()
        d_wall = np.min(np.linalg.norm(wall_pts - self.pose[:2], axis=1))
        d_ped = np.linalg.norm(self.ped - self.pose[:2])
        return float(min(d_wall, d_ped))

    def _reward(self, action, dist):
        r = self.rcfg
        rew = r.progress * (self.prev_dist - dist)
        gb = self._goal_body()
        heading_err = abs(np.arctan2(gb[1], gb[0]))
        rew += r.heading * (1.0 - heading_err / np.pi)
        rew += r.smooth * np.linalg.norm(action - self.prev_action)
        rew += r.time
        min_d = self._min_obstacle_dist()
        if min_d < 0.15:
            return rew + r.collision, True, {'event': 'collision', 'min_dist': min_d}
        if min_d < 0.5:
            rew += r.near_obstacle * (0.5 - min_d) / 0.5
        if dist < 0.3:
            return rew + r.success, True, {'event': 'success', 'min_dist': min_d}
        if self.steps >= self.max_steps:
            return rew, True, {'event': 'timeout', 'min_dist': min_d}
        return rew, False, {'event': 'step', 'min_dist': min_d}


# TODO(ros2-env): implement a ROS2-backed environment that drives the real
# mock_e2e pipeline. It should:
#   - spin a rclpy node, subscribe /bev/tensor, /anchor/odom
#   - publish actions to /cmd_vel_raw (still passing through safety_supervisor)
#   - compute reward from odom progress + /safety/status
# Kept unimplemented intentionally to avoid fabricating ROS behavior offline.
class Ros2ClosedLoopEnv:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Ros2ClosedLoopEnv is a documented stub. Use KinematicBevEnv for "
            "offline PPO, or implement the rclpy bridge described in "
            "docs/training_strategy.md.")
