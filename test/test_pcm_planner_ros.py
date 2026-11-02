"""Live simulator/planner integration; full-size benchmark is opt-in."""

import json
from collections import Counter
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
from autoware_control_msgs.msg import Control
from autoware_planning_msgs.msg import Trajectory
from builtin_interfaces.msg import Time
from crp_msgs.msg import Ego, PathWithTrafficRules, Scenario
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tier4_planning_msgs.msg import PathPointWithLaneId

from plan_predictive_cost_map.planner.config import PlannerConfig
from plan_predictive_cost_map.planner.messages import snapshot_from_messages, trajectory_message
from plan_predictive_cost_map.planner.costmap import fit_costmap
from plan_predictive_cost_map.planner.optimizer import PolynomialMPC


def test_map_frame_and_elevated_trajectory():
    ego = Ego()
    ego.header.frame_id = 'map'
    ego.header.stamp.sec = 100
    ego.pose.pose.position.x = 10.
    ego.pose.pose.position.z = 5.
    ego.pose.pose.orientation.w = 1.
    ego.twist.twist.linear.x = 2.
    scenario = Scenario()
    scenario.header.frame_id = 'base_link'
    scenario.header.stamp = ego.header.stamp
    path = PathWithTrafficRules()
    path.traffic_rules.maximum_speed = 5.
    for distance in (0., 30.):
        point = PathPointWithLaneId()
        point.point.pose.position.x = distance
        point.point.pose.position.z = .05 * distance
        path.path.points.append(point)
        path.path.left_bound.append(Point(x=distance, y=1.925, z=.05 * distance))
        path.path.right_bound.append(Point(x=distance, y=-1.925, z=.05 * distance))
    scenario.paths = [path]
    config = PlannerConfig(horizon=.5, gp_support_points=8, road_support_points=8,
                           solve_budget=.1, maximum_iterations=100, jit=False)
    snapshot = snapshot_from_messages(scenario, ego, [0, 0], 1, config)
    fitted = fit_costmap([], snapshot.road.points, (), snapshot.state[:2], config)
    plan = PolynomialMPC(config).solve(snapshot.state, [0, 0], snapshot.road, fitted)
    trajectory = trajectory_message(plan, snapshot, config)
    assert trajectory.header.frame_id == 'map'
    assert trajectory.header.stamp.sec == 100
    assert len(trajectory.points) == 11
    assert trajectory.points[-1].pose.position.z > 5.
    assert trajectory.points[-1].pose.orientation.y < 0.
    assert trajectory.points[-1].time_from_start.nanosec == 500_000_000
    scenario.header.stamp = Time(sec=101)
    with pytest.raises(ValueError):
        snapshot_from_messages(scenario, ego, [0, 0], 2, config)


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
    try:
        return process.communicate(timeout=10)[0]
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return process.communicate()[0]


@pytest.mark.parametrize('slow_optimizer', [False, True])
def test_live_control_loop_and_stale_input(slow_optimizer):
    os.environ['ROS_DOMAIN_ID'] = str(100 + os.getpid() % 100)
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    root = Path(__file__).resolve().parents[3]
    simulator = root / 'install/simulator/lib/simulator/simulator'
    planner = root / 'install/plan_predictive_cost_map/lib/plan_predictive_cost_map/pcm_planner'
    if not simulator.is_file() or not planner.is_file():
        pytest.skip('Build simulator and plan_predictive_cost_map before the live test')
    scenario_file = Path(os.environ.get('PCM_SCENARIO_FILE', str(root / 'src/scenario_files/occluded_pedestrian_crossing_costmap_data.mat')))
    full = os.environ.get('PCM_FULL_BENCHMARK') == '1'
    rclpy.init()
    node = rclpy.create_node('pcm_planner_regression')
    statuses, trajectories, commands, states, planner_statuses, costmaps = [], [], [], [], [], []
    subscriptions = [
        node.create_subscription(String, '/control/mpc_tracker/status', lambda value: statuses.append(json.loads(value.data)), 10),
        node.create_subscription(String, '/plan/pcm_planner/status', lambda value: planner_statuses.append(json.loads(value.data)), 10),
        node.create_subscription(PointCloud2, '/plan/pcm/costmap/t0', costmaps.append, 1),
        node.create_subscription(Trajectory, '/plan/trajectory', trajectories.append, 10),
        node.create_subscription(Control, '/control/command/control_cmd', lambda value: commands.append((time.monotonic(), value)), 100),
        node.create_subscription(Ego, '/ego', states.append, 10),
    ]
    simulation = subprocess.Popen([str(simulator), '--ros-args', '-p', 'scenario_file:=' + str(scenario_file),
        '-p', 'vehicle_model:=dynamic', '-p', 'command_mode:=acceleration',
        '-p', 'use_internal_controller:=false', '-p', 'use_scenario_speed_when_uncommanded:=false',
        '-p', 'simulation_time_stamps:=true',
        '-p', 'stop_at_scenario_end:=false',
        '-p', 'generate_virtual_objects:=' + os.environ.get('PCM_VIRTUAL_OBJECTS', 'false')], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    arguments = [str(planner), '--ros-args', '-p', 'planning_mode:=combined']
    if slow_optimizer:
        arguments += ['-p', 'jit:=false', '-p', 'solve_budget:=0.02']
    elif not full:
        arguments += ['-p', 'horizon:=1.5', '-p', 'gp_support_points:=8',
                      '-p', 'road_support_points:=8', '-p', 'jit:=false']
    planning = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
    tracking = subprocess.Popen([str(root / 'install/plan_predictive_cost_map/lib/plan_predictive_cost_map/mpc_tracker')],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    outputs = []
    try:
        deadline = time.monotonic() + (240 if full else 40)
        tracking_started = None
        while time.monotonic() < deadline:
            if planning.poll() is not None or tracking.poll() is not None or simulation.poll() is not None:
                raise AssertionError('Planner or simulator exited unexpectedly')
            rclpy.spin_once(node, timeout_sec=.02)
            if statuses and statuses[-1].get('command_valid') and any(message.points for message in trajectories):
                tracking_started = tracking_started or time.monotonic()
            if tracking_started and time.monotonic() - tracking_started > 6:
                break
        assert tracking_started, 'No validated plan; recent statuses: ' + repr(statuses[-5:])
        assert any(command.longitudinal.is_defined_acceleration for _, command in commands)
        assert all(np.isfinite(command.longitudinal.acceleration) for _, command in commands)
        assert any(message.points and message.header.frame_id == 'map' for message in trajectories)
        recent_times = [stamp for stamp, _ in commands if stamp >= tracking_started]
        intervals = np.diff(recent_times)
        assert len(intervals) >= 15
        assert np.mean(intervals) < .075, 'Command timer is blocked by optimization'
        successful = [status for status in statuses if status.get('command_valid')]
        assert all(status['solve_ms'] < 100 for status in successful)
        assert costmaps, 'Fitted GP cost map was not published'
        assert all(message.points for message in trajectories), 'Planner must not clear references on failed replans'
        assert any('objective' in status for status in planner_statuses)
        publishers = {entry.node_name for entry in node.get_publishers_info_by_topic('/control/command/control_cmd')}
        assert publishers == {'mpc_tracker'}
        if slow_optimizer:
            assert any(status.get('total_ms', 0) > 50 for status in planner_statuses), 'Slow solver did not exercise the multirate path'
            assert max(state.twist.twist.linear.x for state in states) > .2, repr((max(state.twist.twist.linear.x for state in states), len(successful), len(statuses), Counter(status.get('reason', status.get('state')) for status in statuses), max(command.longitudinal.acceleration for _, command in commands)))
        assert np.linalg.norm(np.array([states[-1].pose.pose.position.x, states[-1].pose.pose.position.y]) -
                      np.array([states[0].pose.pose.position.x, states[0].pose.pose.position.y])) > .1
        outputs.append(stop_process(simulation))
        cutoff = time.monotonic()
        while time.monotonic() - cutoff < .6:
            rclpy.spin_once(node, timeout_sec=.02)
        assert commands[-1][1].longitudinal.acceleration < 0, 'Stale input must brake'
        assert not statuses[-1].get('command_valid'), 'Stale input must invalidate tracking'
        print('Validated cycles:', len(successful), 'command mean ms:', np.mean(intervals) * 1000,
              'tracking median/p95 ms:', np.percentile([status['solve_ms'] for status in successful], [50, 95]))
    finally:
          outputs.append(stop_process(tracking))
        outputs.append(stop_process(planning))
        outputs.append(stop_process(simulation))
        node.destroy_node()
        rclpy.shutdown()
        for output in outputs:
            if 'Traceback' in output:
                print(output)