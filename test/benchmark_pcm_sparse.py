import argparse
import json
from pathlib import Path
import runpy
import sys
from timeit import default_timer
from types import SimpleNamespace

import casadi as ca
import numpy as np
import rclpy
from crp_msgs.msg import PathWithTrafficRules, Scenario
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters
from tier4_planning_msgs.msg import PathPointWithLaneId



def launch_settings():
    launch_file = Path(__file__).resolve().parents[1] / 'launch/pcm_longitudinal.launch.py'
    description = runpy.run_path(str(launch_file))['generate_launch_description']()
    context = LaunchContext()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    action = next(action for action in description.entities if isinstance(action, Node))
    normalized = next(value for key, value in vars(action).items() if key.endswith('__parameters'))
    return evaluate_parameters(context, normalized)[0]


def scenario_inputs(name, horizon):
    scenario = Scenario()
    path = PathWithTrafficRules()
    path.traffic_rules.maximum_speed = 8.0
    for distance in np.linspace(-10.0, 120.0, 131):
        point = PathPointWithLaneId()
        point.point.pose.position.x = float(distance)
        path.path.points.append(point)
    scenario.paths = [path]
    times = np.linspace(0.0, horizon, 41)
    def moving_object(x, y, label, velocities):
        branches = [PredictionBranch(np.column_stack((x + vx * times, y + vy * times)),
                                     horizon / 40.0, probability)
                    for vx, vy, probability in velocities]
        return ObjectState(x, y, 0.0, 0.0, 0.0, label, predicted_branches=branches)
    pedestrians, vehicles = [], []
    if name != 'clear':
        pedestrians = [moving_object(18.0, -3.6, 7, [(0.0, 1.4, 0.7), (0.0, 0.2, 0.3)])]
        vehicles = [moving_object(40.0, 3.4, 1, [(-8.0, 0.0, 0.8), (-5.0, 0.0, 0.2)])]
    if name == 'capacity':
        pedestrians.append(moving_object(35.0, 3.6, 7, [(0.0, -1.2, 0.6), (0.0, -0.3, 0.4)]))
        vehicles.append(moving_object(65.0, -3.4, 1, [(-10.0, 0.0, 0.7), (-6.0, 0.0, 0.3)]))
    return scenario, pedestrians, vehicles


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {name: float(value) for name, value in zip(
        ('mean', 'p50', 'p95', 'p99', 'max'),
        [np.mean(values), *np.percentile(values, [50, 95, 99]), np.max(values)])}


def layer_errors(node):
    helper = SimpleNamespace(N=node.N, t=node.t, dt=node.dt, max_predicted_paths=node.max_predicted_paths,
                             sample_object_path=node.sample_object_path)
    grid = np.meshgrid(np.linspace(0.0, 45.0, 46), [-3.4, 0.0, 3.4],
                       np.linspace(0.0, node.tau_h, 21), indexing='ij')
    queries = np.column_stack([values.ravel() for values in grid])
    result = {}
    for name, objects, limit, kernel, severity, base, growth in (
        ('pedestrian', node.pedestrians, node.max_pedestrians, node.p_ped,
         node.pedestrian_cost_primitive, node.pedestrian_noise_base, node.pedestrian_noise_growth),
        ('vehicle', node.vehicles, node.max_vehicles, node.p_veh,
         node.vehicle_cost_primitive, node.vehicle_noise_base, node.vehicle_noise_growth),
    ):
        full_points, full_alpha, _ = pcm_node.fit_dynamic_gp(helper, objects, limit, kernel, severity, base, growth)
        points, alpha, metadata = node.fit_dynamic_gp(objects, limit, kernel, severity, base, growth)
        full_kernel = numeric_rbf_kernel(queries, full_points, kernel)
        sparse_kernel = numeric_rbf_kernel(queries, points, kernel)
        error = sparse_kernel @ alpha - full_kernel @ full_alpha
        full_gradient = np.sum(full_kernel[:, :, None] * full_alpha[None, :, None] *
            (full_points[None, :, :] - queries[:, None, :]) / np.asarray(kernel[1:])**2, axis=1)
        sparse_gradient = np.sum(sparse_kernel[:, :, None] * alpha[None, :, None] *
            (points[None, :, :] - queries[:, None, :]) / np.asarray(kernel[1:])**2, axis=1)
        bound = metadata.get('error_bound', 0.0)
        result[name] = {'max_abs_mean_error': float(np.max(np.abs(error))),
                        'max_gradient_error': float(np.max(np.linalg.norm(sparse_gradient - full_gradient, axis=1))),
                        'error_bound': bound, 'minimum_upper_margin': float(np.min(error + bound)),
                        'full_supports': len(full_points), 'optimizer_supports': len(points)}
    return result


def benchmark(mode, supports, cycles):
    settings = {**launch_settings(), 'prediction_model': 'dynamic_light', 'scenario_worker': False,
                'publish_gp_state': False, 'timing_enabled': True}
    if mode == 'sparse':
        settings.update(sparse_gp=True, object_support_points=supports)
    arguments = ['--ros-args']
    for name, value in settings.items():
        if isinstance(value, tuple):
            value = list(value)
        arguments.extend(['-p', name + ':=' + (str(value).lower() if isinstance(value, bool) else str(value))])
    rclpy.init(args=arguments)
    started = default_timer()
    node = pcm_node()
    build_seconds = default_timer() - started
    node.get_logger = lambda: SimpleNamespace(info=lambda *args, **kwargs: None,
        warn=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    records = []
    node.timing_pub = SimpleNamespace(publish=lambda message: records.append(json.loads(message.data)))
    result = {'mode': mode, 'supports_per_path': supports if mode == 'sparse' else node.N,
              'horizon': node.tau_h, 'prediction_steps': node.N,
              'build_seconds': build_seconds, 'cases': {}}
    try:
        for name in ('clear', 'crossing', 'capacity'):
            scenario, pedestrians, vehicles = scenario_inputs(name, node.tau_h)
            node.scenario_callback(scenario)
            node.pedestrians, node.vehicles = pedestrians, vehicles
            node.ego_state.vx = 8.0
            node.last_c_opt = np.zeros(node.M)
            node.light_previous = None
            node.light_multipliers = {}
            node.c_numeric = np.zeros(node.M)
            cold = None
            for index in range(cycles + 1):
                node.replan = True
                node.last_exec = 0.0
                if node.light_previous is not None:
                    node.light_previous_time = default_timer() - node.timer_period
                node.callback()
                if index == 0:
                    cold = records[-1]
            warm = records[-cycles:]
            stage_names = [key for key, value in warm[0].items()
                           if key.endswith('_seconds') and key != 'deadline_seconds'
                           and isinstance(value, (int, float))]
            evaluations = {}
            for function_name in ('nlp_f', 'nlp_g', 'nlp_jac_g', 'nlp_hess_l'):
                function = node.solver.get_function(function_name)
                inputs = {'x': node.last_c_opt, 'p': node.p_numeric, 'lam_f': 1.0,
                          'lam_g': np.ones(node.g.numel())}
                call_args = [inputs[value] for value in function.name_in()]
                function.call(call_args)
                durations = []
                for _ in range(50):
                    started = default_timer()
                    function.call(call_args)
                    durations.append(default_timer() - started)
                evaluations[function_name] = summarize(durations)
            result['cases'][name] = {
                'cold_total_seconds': cold['total_seconds'],
                'accepted_cycles': sum(record['accepted'] is True for record in warm),
                'cycles': cycles, 'deadline_misses': sum(record['deadline_missed'] for record in warm),
                'stages': {key: summarize([record[key] for record in warm]) for key in stage_names},
                'iterations': summarize([record['iterations'] for record in warm]),
                'oracle_evaluations': evaluations, 'gp_errors': layer_errors(node),
                'last_status': node.plan_diagnostics.get('status'),
                'last_command_speed': float(node.command_eval(node.c_numeric, node.p_numeric)),
            }
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return result


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    global numeric_rbf_kernel, ObjectState, PredictionBranch, pcm_node
    from plan_predictive_cost_map.pcm_kernels import numeric_rbf_kernel
    from plan_predictive_cost_map.pcm_longitudinal import ObjectState, PredictionBranch, pcm_node
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'sparse', 'both'], default='both')
    parser.add_argument('--supports', type=int, nargs='+', default=[9])
    parser.add_argument('--cycles', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.cycles < 1:
        parser.error('--cycles must be positive')
    modes = ['full', 'sparse'] if arguments.mode == 'both' else [arguments.mode]
    report = {'method': 'Fixed-input synchronous callbacks; launch settings; one cold and repeated warm solves. '
              'Logging suppressed, no live ROS traffic or closed-loop simulator; timings exclude diagnostics encoding. '
              'Grid errors are sampled checks, not the source of the RKHS bound.',
              'results': [benchmark(mode, supports, arguments.cycles) for mode in modes
                          for supports in (arguments.supports if mode == 'sparse' else [0])]}
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for result in report['results']:
        for name, case in result['cases'].items():
            print(json.dumps({'mode': result['mode'], 'supports': result['supports_per_path'],
                'case': name, 'accepted': case['accepted_cycles'],
                'cycles': case['cycles'], 'misses': case['deadline_misses'],
                'total_ms': {key: value * 1000 for key, value in case['stages']['total_seconds'].items()},
                'gp_errors': case['gp_errors']}, sort_keys=True))


if __name__ == '__main__':
    main()