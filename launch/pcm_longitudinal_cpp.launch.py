from pathlib import Path
import hashlib
import json
import os
import runpy

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters


def generate_launch_description():
    original = runpy.run_path(str(Path(__file__).with_name('pcm_longitudinal.launch.py')))['generate_launch_description']()
    declarations = [action for action in original.entities if isinstance(action, DeclareLaunchArgument)]
    nodes = [action for action in original.entities if isinstance(action, Node)]

    def launch_native(context):
        from plan_predictive_cost_map.pcm_cpp_model import export_model

        normalized = next(value for key, value in vars(nodes[0]).items() if key.endswith('__parameters'))
        settings = evaluate_parameters(context, normalized)[0]
        representation = LaunchConfiguration('velocity_representation').perform(context)
        signature = hashlib.sha256((json.dumps(settings, sort_keys=True) + representation).encode()).hexdigest()[:16]
        cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'pcm_longitudinal_cpp' / signature
        directory = export_model(settings, cache, representation)
        return [Node(package='plan_predictive_cost_map', executable='pcm_longitudinal_cpp',
                     name='pcm_longitudinal_cpp', output='screen', parameters=[{'model_directory': directory}]),
                *nodes[1:]]

    return LaunchDescription([*declarations,
        DeclareLaunchArgument('velocity_representation', default_value='polynomial',
            choices=['polynomial', 'acados_stages'], description='Global polynomial or native acados stage-wise speed commands'),
        OpaqueFunction(function=launch_native)])