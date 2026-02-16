from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    pcm = Node(
            package='plan_predictive_cost_map',
            executable='pcm',
            parameters=[
            {"/pcm/weights/weight_costmap": 0.65}, ],
            name='pcm',
            output='screen'
        )
    
    ld.add_action(pcm)

    return ld
