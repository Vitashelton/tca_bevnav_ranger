#!/usr/bin/env python3
"""Load (bev, goal, action) samples from recorded ROS2 bags.

Reads the topics produced by the closed-loop stack:
  /bev/tensor        (e2e_nav_msgs/BevTensor)  -> input
  /goal/vector       (e2e_nav_msgs/GoalVector) -> input
  /cmd_vel_safe      (geometry_msgs/Twist)     -> BC target (executed command)

The expert action for behavior cloning is the *safe* command actually sent to
the base (teleop or teacher passes through safety_supervisor during recording).

This loader uses rosbag2_py + rclpy serialization, which are only available
inside a sourced ROS2 environment. When they are missing we raise a clear
error instead of fabricating data. A pre-extracted .npz path is the
recommended offline interchange (see inspect_dataset.py / bev_dataset.py).
"""
import numpy as np

try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    _HAS_ROS = True
except Exception:
    _HAS_ROS = False


def _require_ros():
    if not _HAS_ROS:
        raise RuntimeError(
            "rosbag2_py / rclpy not found. Source your ROS2 workspace "
            "(source ros2_ws/install/setup.bash) before using rosbag_loader, "
            "or pre-extract bags to .npz with scripts/inspect_dataset.py.")


def nearest(ts_list, t):
    idx = int(np.argmin([abs(x - t) for x in ts_list]))
    return idx


def load_bag(bag_path, max_dt_ns=80_000_000):
    """Return list of dict(bev, goal, action). Nearest-time matched to BEV."""
    _require_ros()
    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag_path, storage_id='sqlite3'),
                ConverterOptions('', ''))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    bevs, goals, cmds = [], [], []
    while reader.has_next():
        topic, data, t = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        if topic == '/bev/tensor':
            arr = np.asarray(msg.data, np.float32).reshape(
                msg.channels, msg.height, msg.width)
            bevs.append((t, arr))
        elif topic == '/goal/vector':
            goals.append((t, np.array(
                [msg.dx, msg.dy, msg.distance, msg.heading_error], np.float32)))
        elif topic == '/cmd_vel_safe':
            cmds.append((t, np.array(
                [msg.linear.x, msg.linear.y, msg.angular.z], np.float32)))
    samples = []
    g_ts = [g[0] for g in goals]
    c_ts = [c[0] for c in cmds]
    for t, bev in bevs:
        if not g_ts or not c_ts:
            break
        gi, ci = nearest(g_ts, t), nearest(c_ts, t)
        if abs(g_ts[gi] - t) > max_dt_ns or abs(c_ts[ci] - t) > max_dt_ns:
            continue
        samples.append({'bev': bev, 'goal': goals[gi][1], 'action': cmds[ci][1]})
    return samples
