from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('pcl_tutorial')
    config = os.path.join(pkg_share, 'config', 'pcl_tutorial.yaml')
    default_rviz = os.path.join(pkg_share, 'rviz', 'Recorded-Can_filtered_segmented.rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_rviz',
            default_value='false',
            description='Launch RViz2 with package RViz config',
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=default_rviz,
            description='Absolute path to RViz2 config file',
        ),
        Node(
            package='pcl_tutorial',
            executable='pcl_tutorial_node',
            name='pcl_tutorial_node',
            output='screen',
            parameters=[config],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ),
    ])
