from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='perceptors',
            executable='camera_perceptor',
            name='camera_perceptor',
            output='screen'
        ),
        Node(
            package='perceptors',
            executable='lidar_perceptor',
            name='lidar_perceptor',
            output='screen'
        )
    ])
