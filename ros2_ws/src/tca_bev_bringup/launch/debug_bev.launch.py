#!/usr/bin/env python3
"""Debug BEV with real sensors: adapters -> time_align -> anchor -> calib -> bev -> viz."""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('fusion_mode', default_value='tca'),
        Node(package='mid360s_adapter', executable='mid360s_adapter_node',
             name='mid360s_adapter', output='screen',
             parameters=[{'mock_mode': False}]),
        Node(package='d435i_adapter', executable='d435i_adapter_node',
             name='d435i_adapter', output='screen',
             parameters=[{'mock_mode': False}]),
        Node(package='time_align', executable='time_align_node', name='time_align',
             parameters=[{'sync_anchor': 'lidar', 'odom_topic': '/anchor/odom'}]),
        Node(package='pose_anchor_manager', executable='pose_anchor_manager_node',
             name='pose_anchor_manager', parameters=[{'backend_type': 'wheel_odom'}]),
        Node(package='calibration_uncertainty_manager',
             executable='calibration_uncertainty_manager_node',
             name='calibration_uncertainty_manager'),
        Node(package='goal_manager', executable='goal_manager_node', name='goal_manager'),
        Node(package='tca_bev_fusion', executable='tca_bev_fusion_node',
             name='tca_bev_fusion', output='screen',
             parameters=[{'fusion_mode': LaunchConfiguration('fusion_mode')}]),
        Node(package='bev_visualization', executable='bev_visualization_node',
             name='bev_visualization'),
    ])
