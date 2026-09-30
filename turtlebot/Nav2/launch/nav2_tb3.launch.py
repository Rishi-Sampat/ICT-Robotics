from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    tb3_nav2_dir = get_package_share_directory('turtlebot3_navigation2')
    nav2_params = os.path.join(
        get_package_share_directory('nav2_setup'), 'params', 'nav2_params.yaml')
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(tb3_nav2_dir, 'launch', 'navigation_launch.py')
            ),
            launch_arguments={
                'params_file': nav2_params,
                'use_sim_time': 'true'
            }.items(),
        )
    ])
