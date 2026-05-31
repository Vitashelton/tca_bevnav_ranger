#!/usr/bin/env python3
"""Real-robot edge stack for Jetson Orin Nano.

Assumes Livox driver and realsense2_camera are launched separately (or add
them here). This file launches the TCA-BEVNav processing chain against real
sensor topics. Set anchor backend via 'anchor_backend' argument.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    anchor = LaunchConfiguration('anchor_backend')
    return LaunchDescription([
        DeclareLaunchArgument('anchor_backend', default_value='wheel_odom'),
        # TODO: include livox_ros_driver2 and realsense2_camera launches here.
        Node(package='mid360s_adapter', executable='mid360s_adapter_node',
             name='mid360s_adapter', output='screen',
             parameters=[{'mock_mode': False}]),
        Node(package='d435i_adapter', executable='d435i_adapter_node',
             name='d435i_adapter', output='screen',
             parameters=[{'mock_mode': False}]),
        Node(package='time_align', executable='time_align_node',
             name='time_align', output='screen',
             parameters=[{'sync_anchor': 'lidar', 'odom_topic': '/anchor/odom'}]),
        Node(package='pose_anchor_manager', executable='pose_anchor_manager_node',
             name='pose_anchor_manager', output='screen',
             parameters=[{'backend_type': anchor}]),
        Node(package='calibration_uncertainty_manager',
             executable='calibration_uncertainty_manager_node',
             name='calibration_uncertainty_manager', output='screen'),
        Node(package='tca_bev_fusion', executable='tca_bev_fusion_node',
             name='tca_bev_fusion', output='screen',
             parameters=[{'fusion_mode': 'tca'}]),
        Node(package='goal_manager', executable='goal_manager_node',
             name='goal_manager', output='screen'),
        Node(package='e2e_policy_runtime', executable='e2e_policy_runtime_node',
             name='e2e_policy_runtime', output='screen',
             parameters=[{'runtime_type': 'tensorrt'}]),
        Node(package='safety_supervisor', executable='safety_supervisor_node',
             name='safety_supervisor', output='screen'),
        Node(package='ranger_base_bridge', executable='ranger_base_bridge_node',
             name='ranger_base_bridge', output='screen',
             parameters=[{'mock_mode': False}]),
    ])
