import argparse
import json
from pathlib import Path
import runpy
import sys
from timeit import default_timer
from types import SimpleNamespace

import numpy as np
import rclpy
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plan_predictive_cost_map.pcm_longitudinal import pcm_node


class TimedSolver:
    def __init__(self, solver, label, calls):
        self.solver, self.label, self.calls = solver, label, calls

    def __getattr__(self, name):
        return getattr(self.solver, name)

    def __call__(self, **arguments):
        started = default_timer()
        result = self.solver(**arguments)
        stats = self.solver.stats()
        self.calls.append({'solver': self.label, 'ms': 1000 * (default_timer() - started),
            'status': stats.get('return_status'), 'iterations': stats.get('iter_count'),
            'maximum_constraint': float(np.max(result['g'].full())),
            'oracles_ms': {key: 1000 * value for key, value in stats.items() if key.startswith('t_wall_nlp_')}})
        if self.label == 'primary':
            coefficients = result['x'].full().flatten()
            outputs = self.solver.get_function('nlp_jac_fg')(coefficients, arguments['p'])
            gradient, jacobian = outputs[1].full().flatten(), outputs[3].full()
            numeric_gradient, numeric_jacobian = [], []
            values = self.solver.get_function('nlp_fg')
            for index in range(len(coefficients)):
                increment = np.zeros_like(coefficients)
                increment[index] = 1e-5 * max(1.0, abs(coefficients[index]))
                upper = values(coefficients + increment, arguments['p'])
                lower = values(coefficients - increment, arguments['p'])
                numeric_gradient.append(float(upper[0] - lower[0]) / (2 * increment[index]))
                numeric_jacobian.append((upper[1] - lower[1]).full().flatten() / (2 * increment[index]))
            self.calls[-1].update(coefficients=coefficients.tolist(), gradient=gradient.tolist(),
                finite_difference_gradient=numeric_gradient,
                jacobian_maximum_error=float(np.max(np.abs(jacobian - np.asarray(numeric_jacobian).T))),
                active_constant_rows=int(np.sum((np.abs(result['g'].full().flatten()) < 1e-8) &
                                               (np.max(np.abs(jacobian), axis=1) < 1e-10))))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', type=Path)
    parser.add_argument('--times', type=float, nargs='+', default=[3.5, 6.0, 6.65, 7.3, 10.0, 15.0])
    parser.add_argument('--output', type=Path)
    arguments = parser.parse_args()
    fixtures = runpy.run_path(str(Path(__file__).with_name('benchmark_pcm_sparse.py')))
    options = {**fixtures['launch_settings'](), 'scenario_worker': False, 'publish_gp_state': False,
               'timing_enabled': True}
    ros_arguments = ['--ros-args']
    for name, value in options.items():
        if isinstance(value, tuple):
            value = list(value)
        ros_arguments += ['-p', name + ':=' + (json.dumps(value) if value != '' else "''")]
    rclpy.init(args=ros_arguments)
    node = pcm_node()
    calls, records, result = [], [], []
    node.get_logger = lambda: SimpleNamespace(info=lambda *args, **kwargs: None,
        warn=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    node.timing_pub = SimpleNamespace(publish=lambda message: records.append(json.loads(message.data)))
    node.solver = TimedSolver(node.solver, 'primary', calls)
    if node.actuators_enabled and node.actuator_method == 'sqp':
        node.light_warm_solver = node.solver
        node.actuator_retry_solver = TimedSolver(node.actuator_retry_solver, 'retry', calls)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(arguments.bag.resolve()), storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {topic.name: get_message(topic.type) for topic in reader.get_all_topics_and_types()}
    topics = ['/ego', '/scenario', '/simulator/actuator_state']
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    origin, scenario, actuator = None, None, None
    pending = iter(sorted(arguments.times))
    requested = next(pending, None)
    try:
        while reader.has_next() and requested is not None:
            topic, raw, stamp = reader.read_next()
            message = deserialize_message(raw, types[topic])
            if topic == '/ego':
                node.ego_callback(message)
                origin = stamp if origin is None else origin
            elif topic == '/scenario':
                scenario = message
            else:
                actuator = message
            if origin is None or (stamp - origin) * 1e-9 < requested or scenario is None or actuator is None:
                continue
            node.scenario_callback(scenario)
            node.powertrain_callback(actuator)
            node.replan, node.last_exec = True, 0.0
            if node.light_previous is not None:
                node.light_previous_time = default_timer() - node.timer_period
            calls.clear()
            node.callback()
            record = {'time_seconds': requested, 'speed': node.ego_state.vx,
                      'diagnostics': dict(node.plan_diagnostics), 'calls': list(calls), 'timing': records[-1]}
            result.append(record)
            print(json.dumps(record, allow_nan=False), flush=True)
            requested = next(pending, None)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if arguments.output:
        arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()