import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import tempfile
import time

import pytest
import rclpy
from autoware_control_msgs.msg import Control
from crp_msgs.msg import Ego
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy

from plan_predictive_cost_map.pcm_cpp_model import export_model


@pytest.mark.skipif(os.environ.get('PCM_NATIVE_LIVE_TEST') != '1', reason='Opt-in non-recorded simulator test')
@pytest.mark.parametrize('representation', ['polynomial', 'acados_stages'])
def test_native_planner_publishes_accepted_commands(tmp_path, representation):
    root = Path(__file__).resolve().parents[3]
    settings = runpy.run_path(str(Path(__file__).with_name('benchmark_pcm_sparse.py')))['launch_settings']()
    if representation == 'acados_stages':
        settings.update(lookahead_time=0.5, prediction_steps=5, object_support_points=5)
    domain = os.environ.get('ROS_DOMAIN_ID')
    os.environ['ROS_DOMAIN_ID'] = '186'
    processes = []
    try:
        export_model(settings, tmp_path, representation)
        rclpy.init(args=[])
        probe = rclpy.create_node('native_live_probe')
        commands, egos, cycles = [], [], []
        subscriptions = [
            probe.create_subscription(Control, '/control/command/control_cmd', commands.append, 100),
            probe.create_subscription(Ego, '/ego', egos.append, 100),
            probe.create_subscription(String, '/plan/pcm_longitudinal/timing',
                lambda message: cycles.append(json.loads(message.data)),
                QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)),
        ]
        with tempfile.TemporaryFile(mode='w+') as log:
            processes.append(subprocess.Popen([
                str(root / 'install/plan_predictive_cost_map/lib/plan_predictive_cost_map/pcm_longitudinal_cpp'),
                '--ros-args', '-p', 'model_directory:=' + str(tmp_path)],
                stdout=log, stderr=log, start_new_session=True))
            deadline = time.monotonic() + 30
            while not any(endpoint.node_name == 'pcm_longitudinal_cpp'
                          for endpoint in probe.get_subscriptions_info_by_topic('/scenario')):
                rclpy.spin_once(probe, timeout_sec=0.05)
                if processes[0].poll() is not None or time.monotonic() > deadline:
                    log.seek(0)
                    pytest.fail('Native startup failed: ' + log.read()[-8000:])
            processes.append(subprocess.Popen(['ros2', 'launch', 'simulator', 'simulator.launch.py'],
                stdout=log, stderr=log, start_new_session=True))
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                rclpy.spin_once(probe, timeout_sec=0.05)
                if any(process.poll() is not None for process in processes):
                    log.seek(0)
                    pytest.fail('Native run exited: ' + log.read()[-8000:])
            log.seek(0)
            assert commands and egos and cycles, log.read()[-8000:]
            assert sum(record['accepted'] for record in cycles) > 20
            if representation == 'acados_stages':
                assert all(record['solver_backend'] == 'acados_sqp' for record in cycles)
                assert all(len(record['speed_command_points']) == settings['prediction_steps'] for record in cycles)
                assert all(record['maximum_constraint'] <= 2 * settings['solver_tolerance']
                           for record in cycles if record['accepted'])
            print('Native non-recorded run:', {'commands': len(commands), 'cycles': len(cycles),
                  'representation': representation,
                  'accepted': sum(record['accepted'] for record in cycles),
                  'mean_cycle_ms': 1000 * sum(record['total_seconds'] for record in cycles) / len(cycles)})
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        if rclpy.ok():
            probe.destroy_node()
            rclpy.shutdown()
        if domain is None:
            os.environ.pop('ROS_DOMAIN_ID', None)
        else:
            os.environ['ROS_DOMAIN_ID'] = domain