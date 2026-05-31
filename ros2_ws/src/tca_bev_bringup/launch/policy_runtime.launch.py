#!/usr/bin/env python3
"""Policy + safety only (assumes BEV / goal provided elsewhere)."""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('runtime_type', default_value='dummy'),
        Node(package='e2e_policy_runtime', executable='e2e_policy_runtime_node',
             name='e2e_policy_runtime', output='screen',
             parameters=[{'runtime_type': LaunchConfiguration('runtime_type')}]),
        Node(package='safety_supervisor', executable='safety_supervisor_node',
             name='safety_supervisor', output='screen'),
    ])
