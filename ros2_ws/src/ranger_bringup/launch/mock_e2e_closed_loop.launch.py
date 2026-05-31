#!/usr/bin/env python3
"""Full mock end-to-end closed loop (M11).

mock_sensors -> time_align -> pose_anchor_manager(mock_odom)
            -> calibration_uncertainty_manager -> tca_bev_fusion
            -> goal_manager -> e2e_policy_runtime(dummy)
            -> safety_supervisor -> ranger_base_bridge(mock) + bev_visualization

Run:
  ros2 launch ranger_bringup mock_e2e_closed_loop.launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    scenario = LaunchConfiguration('scenario')
    fusion_mode = LaunchConfiguration('fusion_mode')

    return LaunchDescription([
        DeclareLaunchArgument('scenario', default_value='corridor_door'),
        DeclareLaunchArgument('fusion_mode', default_value='tca'),

        Node(package='mock_sensors', executable='mock_sensors_node',
             name='mock_sensors', output='screen',
             parameters=[{'scenario': scenario, 'goal_x': 4.0, 'goal_y': 0.0}]),

        Node(package='time_align', executable='time_align_node',
             name='time_align', output='screen',
             parameters=[{'sync_anchor': 'lidar', 'odom_topic': '/mock/odom'}]),

        Node(package='pose_anchor_manager', executable='pose_anchor_manager_node',
             name='pose_anchor_manager', output='screen',
             parameters=[{'backend_type': 'mock_odom', 'expected_freq': 50.0}]),

        Node(package='calibration_uncertainty_manager',
             executable='calibration_uncertainty_manager_node',
             name='calibration_uncertainty_manager', output='screen',
             parameters=[{'camera_extrinsic_confidence': 0.4}]),

        Node(package='tca_bev_fusion', executable='tca_bev_fusion_node',
             name='tca_bev_fusion', output='screen',
             parameters=[{'fusion_mode': fusion_mode, 'publish_debug_image': True}]),

        Node(package='goal_manager', executable='goal_manager_node',
             name='goal_manager', output='screen',
             parameters=[{'goal_topic': '/goal/raw', 'odom_topic': '/anchor/odom'}]),

        Node(package='e2e_policy_runtime', executable='e2e_policy_runtime_node',
             name='e2e_policy_runtime', output='screen',
             parameters=[{'runtime_type': 'dummy'}]),

        Node(package='safety_supervisor', executable='safety_supervisor_node',
             name='safety_supervisor', output='screen'),

        Node(package='ranger_base_bridge', executable='ranger_base_bridge_node',
             name='ranger_base_bridge', output='screen',
             parameters=[{'mock_mode': True}]),

        Node(package='bev_visualization', executable='bev_visualization_node',
             name='bev_visualization', output='screen'),
    ])
