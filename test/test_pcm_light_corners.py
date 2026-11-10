import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import numpy as np
import pytest
import rclpy
from autoware_control_msgs.msg import Control
from crp_msgs.msg import Ego
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


@pytest.mark.skipif(os.environ.get('PCM_LIGHT_CORNERS_TEST') != '1', reason='Opt-in non-recorded corner regression')
@pytest.mark.parametrize('model', ['kinematic', 'dynamic_light'])
@pytest.mark.parametrize('scenario', ['roundabout_with_traffic_costmap_data.mat',
                                    'roundabout_with_conflicting_traffic_costmap_data.mat'])
def test_light_models_roundabout_without_timeout_braking(model, scenario):
    root = Path(__file__).resolve().parents[3]
    previous_domain = os.environ.get('ROS_DOMAIN_ID')
    os.environ['ROS_DOMAIN_ID'] = os.environ.get('PCM_LIGHT_CORNER_DOMAIN', str(195 + (0 if model == 'kinematic' else 1)))
    rclpy.init(args=[])
    probe = rclpy.create_node('light_corner_probe')
    egos, commands, timings, snapshots = [], [], [], []
    subscriptions = [
        probe.create_subscription(Ego, '/ego', lambda message: egos.append((time.monotonic(), message)), 100),
        probe.create_subscription(Control, '/control/command/control_cmd', lambda message: commands.append((time.monotonic(), message)), 100),
        probe.create_subscription(String, '/plan/pcm_longitudinal/timing',
            lambda message: timings.append(json.loads(message.data)),
            QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)),
        probe.create_subscription(String, '/plan/pcm_longitudinal/gp_state',
            lambda message: snapshots.append(json.loads(message.data)),
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)),
    ]
    processes = []
    try:
        with tempfile.TemporaryFile(mode='w+') as log:
            processes.append(subprocess.Popen(['ros2', 'launch', 'plan_predictive_cost_map',
                'pcm_longitudinal.launch.py', 'prediction_model:=' + model],
                stdout=log, stderr=log, start_new_session=True))
            deadline = time.monotonic() + 180
            while not any(endpoint.node_name == 'pcm_longitudinal'
                          for endpoint in probe.get_subscriptions_info_by_topic('/scenario')):
                rclpy.spin_once(probe, timeout_sec=0.05)
                if processes[0].poll() is not None or time.monotonic() > deadline:
                    log.seek(0)
                    pytest.fail('Planner startup failed: ' + log.read()[-6000:])
            processes.append(subprocess.Popen(['ros2', 'launch', 'simulator', 'simulator.launch.py',
                'scenario_file:=' + str(root / 'src/scenario_files' / scenario),
                'initial_displacement:=0.0', 'initial_speed:=13.0'],
                stdout=log, stderr=log, start_new_session=True))
            deadline = time.monotonic() + 45
            start = None
            while time.monotonic() < deadline:
                rclpy.spin_once(probe, timeout_sec=0.05)
                if any(process.poll() is not None for process in processes):
                    log.seek(0)
                    pytest.fail('Corner launch exited: ' + log.read()[-6000:])
                if egos and start is None:
                    start = egos[0][0]
                if start is not None and time.monotonic() - start >= 30:
                    break
            assert commands and timings and snapshots, 'No planner outputs in corner case'
            command_times = np.asarray([sample[0] for sample in commands])
            ego_times = np.asarray([sample[0] for sample in egos])
            indices = np.searchsorted(command_times, ego_times, side='right') - 1
            observed = indices >= 0
            ages = ego_times[observed] - command_times[indices[observed]]
            accepted = sum(record['accepted'] is True for record in timings)
            result = {'model': model, 'scenario': scenario, 'commands': len(commands),
                'accepted': accepted, 'cycles': len(timings),
                'cycle_p50_p95_ms': (np.percentile([record['total_seconds'] for record in timings], [50, 95]) * 1000).tolist(),
                'maximum_command_gap_seconds': float(np.max(np.diff(command_times))),
                'stale_command_samples': int(np.sum(ages > 1.0)),
                'minimum_speed_mps': float(min(message.twist.twist.linear.x for _, message in egos)),
                'maximum_vehicle_error_offset': max(snapshot['layers']['vehicle']['offset'] for snapshot in snapshots)}
            print('CORNER ' + json.dumps(result), flush=True)
            assert accepted >= 0.9 * len(timings), result
            assert result['maximum_command_gap_seconds'] < 1.0, result
            assert result['stale_command_samples'] == 0, result
            assert result['maximum_vehicle_error_offset'] < 0.05, result
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        probe.destroy_node()
        rclpy.shutdown()
        if previous_domain is None:
            os.environ.pop('ROS_DOMAIN_ID', None)
        else:
            os.environ['ROS_DOMAIN_ID'] = previous_domain