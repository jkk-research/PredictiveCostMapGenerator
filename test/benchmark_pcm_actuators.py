import argparse
import json
from pathlib import Path
import runpy
import sys
from timeit import default_timer
from types import SimpleNamespace

import numpy as np
import rclpy


def summary(values):
    values = np.asarray(values, dtype=float)
    return {name: float(value) for name, value in zip(
        ('mean', 'p50', 'p95', 'max'), [np.mean(values), *np.percentile(values, [50, 95]), np.max(values)])}


def detailed_check(node, rollout):
    import casadi as ca

    coefficients = np.asarray(node.c_numeric, dtype=float).flatten()
    ego = node.ego_state
    inputs = [coefficients, [ego.x, ego.y, ego.z, ego.yaw, ego.vx, ego.vy, ego.yawrate],
              ego.ax, ego.steeringAngle, node.c_el, node.c_ez, ego.pitch, node.reference_data,
              node.powertrain_state]
    detailed = rollout(*inputs)
    predicted = node.rollout_eval(coefficients, node.p_numeric)
    _, field = node.cost_sup(node.c_sym, node.p_sym, prediction=detailed)
    field_function = ca.Function('detailed_constraint_check', [node.c_sym, node.p_sym], [field])
    field_values = field_function(coefficients, node.p_numeric).full().flatten()
    acceleration = detailed[2].full()
    speeds = detailed[1].full().flatten()
    violation = max(np.max(field_values - node.max_gp_cost), np.max(np.abs(acceleration[:, 0]) - node.ax_max),
                    np.max(np.abs(acceleration[:, 1]) - node.ay_max), np.max(-speeds),
                    np.max(speeds - max(ego.vx, node.target_speed)),
                    np.max(detailed[9].full() - node.reference_data[1]))
    return {'max_position_error_m': float(np.max(np.linalg.norm(predicted[0].full() - detailed[0].full(), axis=1))),
            'max_speed_error_mps': float(np.max(np.abs(predicted[1].full() - detailed[1].full()))),
            'predicted_net_j': float(predicted[12]), 'detailed_net_j': float(detailed[12]),
            'net_error_j': float(predicted[12] - detailed[12]),
            'drawn_error_j': float(predicted[10] - detailed[10]),
            'recovered_error_j': float(predicted[11] - detailed[11]),
            'detailed_maximum_constraint': float(violation),
            'detailed_feasible': bool(violation <= node.solver_tolerance * 2.0)}


def benchmark(arguments, variant, node_type, fixtures, rollout_builder, case_names):
    options = {**fixtures['launch_settings'](), 'prediction_model': 'dynamic_actuators',
               'lookahead_time': arguments.horizon, 'prediction_steps': arguments.steps,
               'sparse_gp': False, 'timing_enabled': True, 'scenario_worker': False,
               'publish_gp_state': False, 'actuator_integration': 'discrete' if variant == 'discrete' else 'detailed',
               'actuator_stage_step': arguments.stage_step,
               'actuator_solver.hessian': 'exact' if variant == 'baseline' else 'limited-memory',
               'actuator_solver.jit': variant != 'baseline'}
    if arguments.hessian != 'auto':
        options['actuator_solver.hessian'] = arguments.hessian
    if arguments.efficiency_model:
        options['actuator_efficiency.model'] = arguments.efficiency_model
    if arguments.method:
        options['actuator_solver.method'] = arguments.method
    ros_arguments = ['--ros-args']
    for name, value in options.items():
        if isinstance(value, tuple):
            value = list(value)
        encoded = str(value).lower() if isinstance(value, bool) else str(value)
        if value == '':
            encoded = "''"
        ros_arguments.extend(['-p', name + ':=' + encoded])
    rclpy.init(args=ros_arguments)
    started = default_timer()
    node = node_type()
    build_seconds = default_timer() - started
    node.get_logger = lambda: SimpleNamespace(info=lambda *args, **kwargs: None,
        warn=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    records = []
    node.timing_pub = SimpleNamespace(publish=lambda message: records.append(json.loads(message.data)))
    result = {'variant': variant, 'build_seconds': build_seconds, 'horizon': node.tau_h,
              'prediction_steps': node.N, 'stage_step': arguments.stage_step,
              'hessian': options['actuator_solver.hessian'],
              'efficiency_model': node.efficiency_model, 'method': node.actuator_method, 'cases': {}}
    try:
        efficiency_module = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                              'plan_predictive_cost_map/pcm_powertrain.py'))
        detailed_rollout = rollout_builder(node.tau_h, node.N, node.model_control_step,
            node.command_lookahead_time, node.vehicle_settings, node.use_road_gradient, node.model_road_pitch,
            node.reference_samples, node.powertrain_settings,
            efficiency_module['motor_efficiency_function'](node.motor_efficiency_map), node.actuator_vehicle_model)
        for case in case_names:
            fixture_case = 'clear' if case in ('rest', 'moving') else case
            scenario, pedestrians, vehicles = fixtures['scenario_inputs'](fixture_case, node.tau_h)
            node.scenario_callback(scenario)
            node.pedestrians, node.vehicles = pedestrians, vehicles
            node.ego_state.vx = 0.0 if case == 'rest' else 8.0
            node.powertrain_state = np.array([node.ego_state.vx * 9.0 / 0.32, 0.0, 0.0, 0.0])
            node.c_numeric = np.zeros(4)
            node.last_c_opt = np.zeros(4)
            node.light_previous = None
            node.light_multipliers = {}
            records.clear()
            checks = []
            for index in range(arguments.cycles + 1):
                node.replan = True
                node.last_exec = 0.0
                node.powertrain_last_update = default_timer()
                if node.light_previous is not None:
                    node.light_previous_time = default_timer() - node.timer_period
                node.callback()
                accepted = records[-1]['accepted'] is True
                checks.append(detailed_check(node, detailed_rollout) if accepted else None)
            warm = records[1:]
            keys = ('total_seconds', 'optimizer_seconds', 'initialization_seconds', 'postprocess_seconds')
            result['cases'][case] = {
                'cold': records[0], 'cold_detailed_check': checks[0],
                'warm': {key: summary([record[key] for record in warm]) for key in keys},
                'warm_accepted': sum(record['accepted'] is True for record in warm),
                'warm_deadline_misses': sum(record['deadline_missed'] for record in warm),
                'warm_cycles': len(warm), 'warm_iterations': summary([record['iterations'] for record in warm]),
                'warm_detailed_checks': checks[1:], 'last_status': node.plan_diagnostics.get('status')}
            print(json.dumps({'variant': variant, 'horizon': node.tau_h, 'case': case,
                'hessian': options['actuator_solver.hessian'], 'efficiency_model': node.efficiency_model,
                'build_s': build_seconds, 'cold_solver_ms': records[0]['optimizer_seconds'] * 1000.0,
                'warm_solver_ms': result['cases'][case]['warm']['optimizer_seconds']['p50'] * 1000.0,
                'warm_total_ms': result['cases'][case]['warm']['total_seconds']['p50'] * 1000.0,
                'accepted': result['cases'][case]['warm_accepted'], 'cycles': len(warm),
                'last_detailed_check': checks[-1]}, sort_keys=True), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return result


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from plan_predictive_cost_map.pcm_longitudinal import ObjectState, PredictionBranch, pcm_node
    from plan_predictive_cost_map.pcm_longitudinal_dynamics import build_dynamic_rollout

    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', choices=['baseline', 'solver_only', 'discrete'],
                        default=['baseline', 'solver_only', 'discrete'])
    parser.add_argument('--horizon', type=float, default=0.5)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--stage-step', type=float, default=0.005)
    parser.add_argument('--hessian', choices=['auto', 'exact', 'limited-memory'], default='auto')
    parser.add_argument('--efficiency-model', choices=['lut', 'polynomial', 'spline'])
    parser.add_argument('--method', choices=['ipopt', 'sqp'])
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--cases', nargs='+', choices=['rest', 'moving', 'crossing', 'capacity'], default=['rest', 'moving'])
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.cycles < 1:
        parser.error('--cycles must be positive')
    fixtures = runpy.run_path(str(Path(__file__).with_name('benchmark_pcm_sparse.py')),
                             init_globals={'ObjectState': ObjectState, 'PredictionBranch': PredictionBranch})
    report = {'method': 'Fixed measured states; full GP; current launch physical/objective settings; '
              'one cold and repeated warm solves. JIT compilation is reported separately. '
              'Detailed-model replay checks are outside timings; no closed-loop or live deadline guarantee.',
              'results': []}
    for variant in arguments.variants:
        report['results'].append(benchmark(arguments, variant, pcm_node, fixtures,
                                           build_dynamic_rollout, arguments.cases))
        arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()