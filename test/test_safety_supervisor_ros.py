"""ROS graph contracts and independent safety supervision checks."""

import pytest
import numpy as np
from types import SimpleNamespace
from time import monotonic

rclpy = pytest.importorskip('rclpy')
from autoware_control_msgs.msg import Control
from crp_msgs.msg import Ego
from plan_predictive_cost_map.planner.config import SafetyConfig, VehicleConfig
from plan_predictive_cost_map.planner.costmap import Branch
from plan_predictive_cost_map.planner.dynamics import acceleration_command
from plan_predictive_cost_map.planner.tracking import TrackingReference
from plan_predictive_cost_map.safety_supervisor import SafetySupervisor, intersecting_hazard


def test_tracker_subscribes_only_to_ego_and_trajectory_and_publishes_private_commands():
    from plan_predictive_cost_map.mpc_tracker import MPCTracker
    rclpy.init()
    node = MPCTracker()
    try:
        inputs = {subscription.topic_name for subscription in node.subscriptions}
        outputs = {publisher.topic_name for publisher in node.publishers}
        assert '/ego' in inputs and '/plan/trajectory' in inputs
        assert '/scenario' not in inputs
        assert '/simulator/tracking_scenario' not in inputs
        assert '/control/tracking/control_cmd' in outputs
        assert '/control/command/control_cmd' not in outputs
    finally:
        node.destroy_node()
        rclpy.shutdown()


def reference_and_snapshot(branches=()):
    times = np.arange(0., 3.025, .05)
    states = np.zeros((len(times), 7))
    states[:, 0] = 5 * times
    states[:, 4] = 5.
    reference = TrackingReference(0, times, states, np.zeros(len(times)), np.zeros(len(times)), np.zeros(len(times)))
    snapshot = SimpleNamespace(state=states[0], stamp_ns=0, received_at=monotonic(), branches=branches)
    return reference, snapshot


def test_safety_guard_distinguishes_real_and_probability_filtered_virtual_hazards():
    times = np.array([0., 3.])
    real = Branch(np.array([[6., 0., 0.], [6., 0., 0.]]), times, 1., 1., 0, (4., 2., 1.5))
    possible = Branch(real.points, times, .9, .02, 1, (.6, .6, 1.8), True)
    reference, snapshot = reference_and_snapshot([real])
    assert intersecting_hazard(snapshot, reference, 0, SafetyConfig()) == 'real'
    _, snapshot = reference_and_snapshot([possible])
    assert intersecting_hazard(snapshot, reference, 0, SafetyConfig()) is None


def test_supervisor_passes_fresh_commands_and_brakes_stale_candidates():
    rclpy.init()
    node = SafetySupervisor()
    try:
        now = monotonic()
        node._vehicle = VehicleConfig()
        node._drive_command = acceleration_command(node._vehicle)
        ego = Ego()
        ego.header.frame_id = 'map'
        ego.pose.pose.orientation.w = 1.
        ego.twist.twist.linear.x = 5.
        node._ego = ego
        node._ego_received = now
        node._scenario_received = now
        node._reference, node._snapshot = reference_and_snapshot()
        node._reference_received = now
        candidate = Control()
        candidate.longitudinal.is_defined_acceleration = True
        candidate.longitudinal.acceleration = .4
        candidate.longitudinal.velocity = 5.
        node._candidate = candidate
        node._command_received = now
        command, status = node._select_command(now)
        assert status['state'] == 'pass_through'
        assert command.longitudinal.acceleration == pytest.approx(.4)
        assert command is not candidate
        node._command_received = now - 1.
        command, status = node._select_command(now)
        first = status['requested_net_acceleration']
        assert first < 0 and status['intervening']
        command, status = node._select_command(now)
        assert status['requested_net_acceleration'] < first
        outputs = {publisher.topic_name for publisher in node.publishers}
        assert '/control/command/control_cmd' in outputs
    finally:
        node.destroy_node()
        rclpy.shutdown()