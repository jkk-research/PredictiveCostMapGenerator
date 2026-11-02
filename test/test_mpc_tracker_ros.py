"""Real simulator/controller checks with no GP planner in the loop."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

rclpy = pytest.importorskip('rclpy')
from autoware_control_msgs.msg import Control
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from crp_msgs.msg import Ego
from std_msgs.msg import String


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
    try:
        return process.communicate(timeout=10)[0]
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return process.communicate()[0]


def generated_reference(route, ego):
    message = Trajectory()
    message.header.frame_id = 'map'
    message.header.stamp = ego.header.stamp
    if route.shape[1] == 2:
        route = np.column_stack([route, np.zeros(len(route))])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    times = np.arange(0, 5.025, .05)
    points = np.column_stack([np.interp(times * (50 / 3.6), arc, route[:, axis]) for axis in range(3)])
    tangents = np.gradient(points, axis=0)
    for index, (elapsed, point, tangent) in enumerate(zip(times, points, tangents)):
        sample = TrajectoryPoint()
        sample.pose.position.x, sample.pose.position.y, sample.pose.position.z = map(float, point)
        yaw = np.arctan2(tangent[1], tangent[0])
        pitch = np.arctan2(tangent[2], np.linalg.norm(tangent[:2]))
        orientation = Rotation.from_euler('xyz', [0, -pitch, yaw]).as_quat()
        sample.pose.orientation.x, sample.pose.orientation.y, sample.pose.orientation.z, sample.pose.orientation.w = map(float, orientation)
        sample.longitudinal_velocity_mps = 50 / 3.6
        nanoseconds = round(elapsed * 1e9)
        sample.time_from_start.sec = nanoseconds // 1_000_000_000
        sample.time_from_start.nanosec = nanoseconds % 1_000_000_000
        message.points.append(sample)
    return message


def test_real_downhill_tracking_without_planner_and_reference_expiry():
    os.environ['ROS_DOMAIN_ID'] = str(100 + os.getpid() % 100)
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    root = Path(__file__).resolve().parents[3]
    simulator = root / 'install/simulator/lib/simulator/simulator'
    tracker = root / 'install/plan_predictive_cost_map/lib/plan_predictive_cost_map/mpc_tracker'
    if not simulator.is_file() or not tracker.is_file():
        pytest.skip('Build simulator and plan_predictive_cost_map before ROS tests')
    scenario = root / 'src/scenario_files/occluded_pedestrian_costmap_data.mat'
    route = np.asarray(loadmat(scenario, simplify_cells=True)['targetPath_qg'])
    rclpy.init()
    node = rclpy.create_node('independent_tracker_regression')
    states, commands, statuses = [], [], []
    subscriptions = [node.create_subscription(Ego, '/ego', states.append, 100),
        node.create_subscription(Control, '/control/command/control_cmd', lambda value: commands.append((time.monotonic(), value)), 100),
        node.create_subscription(String, '/control/mpc_tracker/status', lambda value: statuses.append(json.loads(value.data)), 100)]
    references = node.create_publisher(Trajectory, '/plan/trajectory', 1)
    simulation = subprocess.Popen([str(simulator), '--ros-args', '-p', 'scenario_file:=' + str(scenario),
        '-p', 'vehicle_model:=dynamic', '-p', 'command_mode:=acceleration',
        '-p', 'use_internal_controller:=false', '-p', 'simulation_time_stamps:=true'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    tracking = subprocess.Popen([str(tracker), '--ros-args', '-p', 'maximum_reference_age:=1.4',
        '-r', '/control/tracking/control_cmd:=/control/command/control_cmd'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    outputs = []
    try:
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
            if len(states) >= 6 and statuses and statuses[-1]['state'] == 'ready' and references.get_subscription_count():
                break
        assert len(states) >= 6
        assert not commands, 'Controller must not brake away the initial state while waiting for a reference'
        assert all(abs(state.twist.twist.linear.x - 50 / 3.6) < 1e-9 for state in states)
        assert all(state.header.stamp.sec == 0 and state.header.stamp.nanosec == 0 for state in states)
        references.publish(generated_reference(route, states[-1]))
        started = time.monotonic()
        while time.monotonic() - started < 2.1:
            if tracking.poll() is not None:
                raise AssertionError('Tracker exited unexpectedly')
            rclpy.spin_once(node, timeout_sec=.02)
        valid = [status for status in statuses if status.get('command_valid')]
        assert len(valid) > 10, statuses[-5:]
        steady = [state for state in states if 0 < state.header.stamp.sec + state.header.stamp.nanosec * 1e-9 < .7]
        assert steady
        assert max(abs(state.twist.twist.linear.x - 50 / 3.6) for state in steady) < .15
        assert any(status['state'] == 'controlled_stop' and 'stale' in status.get('reason', '') for status in statuses)
        assert states[-1].twist.twist.linear.x < 50 / 3.6
        assert all(np.isfinite(command.longitudinal.acceleration) for _, command in commands)
        intervals = np.diff([stamp for stamp, _ in commands])
        assert np.mean(intervals) < .075
        publishers = {entry.node_name for entry in node.get_publishers_info_by_topic('/control/command/control_cmd')}
        assert publishers == {'mpc_tracker'}
        assert any(state.header.stamp.nanosec > 0 for state in states)
        print('Independent downhill tracker command mean ms', np.mean(intervals) * 1000,
              'solve p95 ms', np.percentile([status['solve_ms'] for status in valid], 95))
    finally:
        outputs.append(stop_process(tracking))
        outputs.append(stop_process(simulation))
        node.destroy_node()
        rclpy.shutdown()
        for output in outputs:
            if 'Traceback' in output:
                print(output)


def test_safety_adapter_does_not_build_planner_road_polygons():
    from crp_msgs.msg import Scenario, PathWithTrafficRules
    from geometry_msgs.msg import Point
    from tier4_planning_msgs.msg import PathPointWithLaneId
    from plan_predictive_cost_map.planner.config import PlannerConfig
    from plan_predictive_cost_map.planner.messages import snapshot_from_messages
    ego = Ego()
    ego.header.frame_id = 'map'
    ego.pose.pose.orientation.w = 1.
    scenario = Scenario()
    scenario.header.frame_id = 'base_link'
    corridor = PathWithTrafficRules()
    for distance in (0., 10.):
        point = PathPointWithLaneId()
        point.point.pose.position.x = distance
        corridor.path.points.append(point)
        corridor.path.left_bound.append(Point(x=distance, y=1.925))
        corridor.path.right_bound.append(Point(x=distance, y=-1.925))
    scenario.paths = [corridor]
    snapshot = snapshot_from_messages(scenario, ego, [0, 0], 0, PlannerConfig(), safety_only=True)
    assert snapshot.road.drivable_area is None
    assert len(snapshot.road.points) == 2 and not snapshot.lanes