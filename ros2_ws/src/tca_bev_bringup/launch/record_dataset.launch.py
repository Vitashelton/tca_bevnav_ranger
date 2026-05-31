#!/usr/bin/env python3
"""Write a dataset session metadata file (recording via ros2 bag separately)."""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scenario_name', default_value='corridor_door'),
        Node(package='data_recorder', executable='data_recorder_node',
             name='data_recorder', output='screen',
             parameters=[{'scenario_name': LaunchConfiguration('scenario_name')}]),
    ])
