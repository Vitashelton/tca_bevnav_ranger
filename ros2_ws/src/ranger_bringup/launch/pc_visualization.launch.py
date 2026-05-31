#!/usr/bin/env python3
"""PC-side visualization: RViz2 + bev_visualization."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # rviz config lives in repo rviz/ folder; pass absolute path at runtime if needed
    return LaunchDescription([
        Node(package='rviz2', executable='rviz2', name='rviz2', output='screen'),
        Node(package='bev_visualization', executable='bev_visualization_node',
             name='bev_visualization', output='screen'),
    ])
