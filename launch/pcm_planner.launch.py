from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_nodes(context):
    mode = LaunchConfiguration('planning_mode').perform(context)
    if mode not in ('longitudinal', 'combined'):
        raise ValueError('planning_mode must be longitudinal or combined')
    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('simulator'), 'launch', 'simulator.launch.py'])),
        condition=IfCondition(LaunchConfiguration('start_simulator')),
        launch_arguments={
            'scenario_file': LaunchConfiguration('scenario_file'),
            'command_mode': 'acceleration', 'vehicle_model': 'dynamic',
            'use_internal_controller': 'false',
            'simulation_time_stamps': 'true',
            'use_scenario_speed_when_uncommanded': 'false',
            'initial_speed': '13.88888888888889', 'wait_for_first_control_command': 'true',
        }.items())
    planner = Node(package='plan_predictive_cost_map', executable='pcm_planner', name='pcm_planner',
        output='screen', parameters=[LaunchConfiguration('config_file'), {'planning_mode': mode}])
    tracker = Node(package='plan_predictive_cost_map', executable='mpc_tracker', name='mpc_tracker',
        output='screen', parameters=[LaunchConfiguration('config_file')],
        condition=IfCondition(LaunchConfiguration('start_tracker')))
    safety = Node(package='plan_predictive_cost_map', executable='safety_supervisor', name='safety_supervisor',
        output='screen', parameters=[LaunchConfiguration('config_file')],
        condition=IfCondition(LaunchConfiguration('start_tracker')))
    return [simulator, safety, tracker, planner]


def generate_launch_description():
    configuration = PathJoinSubstitution([
        FindPackageShare('plan_predictive_cost_map'), 'config', 'pcm_planner.yaml'])
    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=configuration),
        DeclareLaunchArgument('planning_mode', default_value='combined'),
        DeclareLaunchArgument('start_tracker', default_value='true'),
        DeclareLaunchArgument('start_simulator', default_value='false'),
        DeclareLaunchArgument('scenario_file', default_value=PathJoinSubstitution([
            FindPackageShare('simulator'), 'scenario_files', 'occluded_pedestrian_costmap_data.mat'])),
        OpaqueFunction(function=launch_nodes),
    ])