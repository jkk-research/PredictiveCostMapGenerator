from pathlib import Path
import json
import math
import os
import runpy
import signal
import subprocess
import time
from types import SimpleNamespace
from timeit import default_timer

import numpy as np
import pytest
import casadi as ca

rclpy = pytest.importorskip('rclpy')
from crp_msgs.msg import PathWithTrafficRules, Scenario
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters
from tier4_planning_msgs.msg import PathPointWithLaneId

from plan_predictive_cost_map.pcm_longitudinal import ObjectState, pcm_node
from plan_predictive_cost_map.pcm_longitudinal_dynamics import shift_velocity_coefficients
from plan_predictive_cost_map.pcm_costmap_viewer import CostmapViewer, evaluate_field, road_height, surface_messages, validate_snapshot


def reference_data(points=None):
    from plan_predictive_cost_map.pcm_lane_geometry import prepare_progress_reference

    return prepare_progress_reference(np.array([[0., 0., 0.], [100., 0., 0.]]) if points is None else points, 96)


def prediction_parameters(node, speed=8.0, target_speed=8.0):
    parameters = np.zeros(node.p_sym.numel())
    parameters[0], parameters[-1] = speed, target_speed
    offset = 13 + 4 * node.pedestrian_support_count + 4 * node.vehicle_support_count
    geometry = np.r_[node.target_gp_points.reshape(-1, order='F'), node.target_gp_alpha,
                     node.drivable_points.reshape(-1, order='F'), node.drivable_alpha, node.reference_data]
    parameters[offset:offset + len(geometry)] = geometry
    return parameters


@pytest.fixture(params=['dynamic', 'kinematic', 'dynamic_light'])
def node(request):
    rclpy.init(args=['--ros-args', '-p', 'lookahead_time:=1.0',
                     '-p', 'prediction_steps:=4',
                     '-p', 'prediction_model:=' + request.param,
                     '-p', 'scenario_worker:=false',
                     '-p', 'vehicle_kernel:=[0.5, 2.0, 3.0, 1.0]'])
    planner = pcm_node()
    try:
        yield planner
    finally:
        planner.destroy_node()
        rclpy.shutdown()


def test_progress_reference_handles_ordered_nonmonotonic_x():
    from plan_predictive_cost_map.pcm_lane_geometry import prepare_progress_reference, progress_reference_function

    points = np.array([[-10., 0., 0.], [0., 0., 0.], [10., 0., 0.],
                       [10., 10., 0.], [0., 10., 0.], [0., 20., 0.]])
    reference = prepare_progress_reference(points, 101)
    lookup = progress_reference_function(101)
    for progress, expected in ((0., [0., 0., 0.]), (5., [5., 0., 0.]),
                                (15., [10., 5., 0.]), (25., [5., 10., 0.]),
                                (35., [0., 15., 0.])):
        np.testing.assert_allclose(lookup(progress, reference).full().flatten()[:3], expected, atol=1e-12)
    query = ca.MX.sym('progress_test')
    derivative = ca.Function('progress_derivative', [query], [ca.jacobian(lookup(query, reference), query)])
    assert np.isfinite(derivative(25.).full()).all()


def test_progress_reference_has_continuous_tangent_at_sample_boundaries():
    from plan_predictive_cost_map.pcm_lane_geometry import prepare_progress_reference, progress_reference_function

    angle = np.linspace(0.0, math.pi / 2.0, 301)
    points = np.column_stack((10.0 * np.sin(angle), 10.0 * (1.0 - np.cos(angle)), np.zeros(len(angle))))
    reference = prepare_progress_reference(points, 96)
    lookup = progress_reference_function(96)
    progress = ca.MX.sym('tangent_progress')
    tangent = ca.Function('reference_tangent', [progress], [ca.jacobian(lookup(progress, reference)[:3], progress)])
    boundary = reference[0] + (reference[1] - reference[0]) * 48.0 / 95.0
    np.testing.assert_allclose(tangent(boundary - 1e-7), tangent(boundary + 1e-7), atol=1e-6, rtol=0.0)


def test_progress_rollout_follows_real_roundabout_without_x_monotonicity():
    from scipy.io import loadmat
    from plan_predictive_cost_map.pcm_longitudinal_dynamics import build_kinematic_rollout

    root = Path(__file__).resolve().parents[2]
    points = np.asarray(loadmat(root / 'scenario_files/roundabout_with_conflicting_traffic_costmap_data.mat',
                                simplify_cells=True)['targetPath_qg'])[:, :2]
    index = 448
    delta = points[index + 1] - points[index]
    yaw = math.atan2(delta[1], delta[0])
    rotation = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    local = (points[index - 80:index + 220] - points[index]) @ rotation.T
    reference = reference_data(np.column_stack((local, np.zeros(len(local)))))
    rollout = build_kinematic_rollout(4.0, 20, 0.25)
    prediction = rollout(ca.DM.zeros(3), ca.DM([0., 0., 0., 0., 5., 0., 0.]), 0., 0.,
                         ca.DM.zeros(4), ca.DM.zeros(4), 0., reference)
    planned = prediction[0].full()[:, :2]
    direction = np.diff(local, axis=0)
    squared_length = np.maximum(np.sum(direction**2, axis=1), 1e-12)
    difference = planned[:, None, :] - local[None, :-1, :]
    fraction = np.clip(np.sum(difference * direction[None, :, :], axis=2) / squared_length, 0.0, 1.0)
    projected = local[None, :-1, :] + fraction[:, :, None] * direction[None, :, :]
    distance = np.min(np.linalg.norm(planned[:, None, :] - projected, axis=2), axis=1)
    assert np.max(distance) < 0.15
    assert planned[-1, 1] < -1.0


@pytest.mark.parametrize('selected_model', ['kinematic', 'dynamic_light', 'dynamic'])
def test_launch_parameters(node, selected_model):
    launch_file = Path(__file__).resolve().parents[1] / 'launch' / 'pcm_longitudinal.launch.py'
    description = runpy.run_path(str(launch_file))['generate_launch_description']()
    context = LaunchContext()
    context.launch_configurations['prediction_model'] = selected_model
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    action = next(action for action in description.entities if isinstance(action, Node))
    normalized = next(value for key, value in vars(action).items() if key.endswith('__parameters'))
    parameters = evaluate_parameters(context, normalized)[0]
    assert parameters['lookahead_time'] == 4.0
    assert parameters['prediction_model'] == selected_model
    assert parameters['prediction_steps'] == 20
    assert parameters['gp_state_period'] == 0.25
    assert parameters['weight_velocity'] == 0.5
    assert parameters['weight_lateral_acceleration'] == 0.5
    assert parameters['weight_crash'] == 1.0
    np.testing.assert_allclose(parameters['pedestrian_kernel'], [1.0, 5.0, 2.5, 0.5])
    assert parameters['dynamic_light.constraint_step'] == 0.1
    assert 'target_speed' not in parameters
    assert set(parameters) <= set(node._parameters)
    assert node.N == 4
    assert node.dt == 0.25
    assert node.target_speed == 0.0
    np.testing.assert_allclose(node.p_veh, [0.5, 2.0, 3.0, 1.0])


def test_planner_inputs_keep_only_latest_sample(node):
    assert node.traj_sub.qos_profile.depth == 1
    assert node.ego_sub.qos_profile.depth == 1


def test_prediction_model_requires_restart(node):
    from rclpy.parameter import Parameter

    other = 'kinematic' if node.prediction_model == 'dynamic' else 'dynamic'
    result = node.set_parameters([Parameter('prediction_model', value=other)])[0]
    assert not result.successful
    assert node.get_parameter('prediction_model').value == node.prediction_model


@pytest.mark.parametrize('pitch,steering', [(0.0, 0.0), (-0.2, 0.0), (0.1, 0.05)])
def test_acceleration_polynomial_drives_simulator_matching_dynamics(pitch, steering):
    from plan_predictive_cost_map.pcm_longitudinal_dynamics import acceleration_polynomial_step
    from plan_predictive_cost_map.planner.dynamics import dynamic_step

    state = ca.DM([0.0, 0.0, 0.0, 0.0, 10.0, 0.1, 0.02])
    coefficients = ca.DM([-1.0, -0.5, 0.1, 0.0])
    elapsed = 0.5
    requested = -1.0 - 0.5 * elapsed + 0.1 * elapsed**2
    actual_state, actual_acceleration, actual_request = acceleration_polynomial_step(0.05)(
        state, coefficients, elapsed, steering, ca.DM([pitch, pitch]))
    expected_state, expected_acceleration = dynamic_step(0.05)(
        state, ca.DM([requested, steering]), ca.DM([pitch, pitch]))
    np.testing.assert_allclose(actual_state.full(), expected_state.full(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_acceleration.full(), expected_acceleration.full(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(float(actual_request), requested, rtol=1e-12)


def test_optimizer_uses_matlab_gp_cost_constraint(node):
    dense_count = node.rollout_eval.size1_out(6)
    expected_count = 8 * node.N + 2 * dense_count
    if node.prediction_model == 'dynamic_light':
        expected_count += 2 * (dense_count - 1)
    assert node.g.numel() == expected_count
    assert node.p_sym.numel() == (27 + 4 * node.pedestrian_support_count + 4 * node.vehicle_support_count
                                 + 3 * node.drivable_support_points + 3 * node.target_support_points
                                 + 2 + 6 * node.reference_samples)
    assert not hasattr(node, 'publish_braking_fallback')
    evaluate = ca.Function('gp_constraint_check', [node.c_sym, node.p_sym], [node.g])
    parameters = np.zeros(node.p_sym.numel())
    parameters[0] = 8.0
    parameters[-1] = 8.0
    actual = evaluate(np.zeros(node.M), parameters).full().flatten()
    field = node.fsup_eval(np.zeros(node.M), parameters)[0].full().flatten()
    np.testing.assert_allclose(actual[4 * node.N:5 * node.N], field - 0.05)


@pytest.mark.parametrize('direction', [-1.0, 1.0])
def test_lateral_acceleration_limit_and_penalty_are_active(node, direction):
    angle = np.linspace(0.0, 1.5, 151)
    points = np.column_stack((10.0 * np.sin(angle), direction * 10.0 * (1.0 - np.cos(angle)),
                              np.zeros(len(angle))))
    node.reference_data = reference_data(points)
    parameters = prediction_parameters(node)
    coefficients = np.zeros(node.M)
    acceleration = node.rollout_eval(coefficients, parameters)[2].full()[:, 1]
    constraints = ca.Function('lateral_constraint_check', [node.c_sym, node.p_sym], [node.g])(
        coefficients, parameters).full().flatten()
    np.testing.assert_allclose(constraints[5 * node.N:6 * node.N], acceleration - node.ay_max)
    np.testing.assert_allclose(constraints[6 * node.N:7 * node.N], -acceleration - node.ay_max)
    assert np.max(np.abs(acceleration)) > node.ay_max
    assert np.max(constraints[5 * node.N:7 * node.N]) > 0.0

    safety, field = node.cost_sup(node.c_sym, node.p_sym)
    velocity = node.get_cost_vel(node.c_sym, node.p_sym)
    base_cost = (node.get_parameter('weight_safety').value * safety
                 + node.get_parameter('weight_velocity').value * velocity
                 + node.get_parameter('weight_crash').value * ca.sum1(ca.exp(5 * node.softplus(field)) - 1))
    base = float(ca.Function('non_lateral_objective', [node.c_sym, node.p_sym], [base_cost])(
        coefficients, parameters))
    objective = float(node.solver.get_function('nlp_f')(coefficients, parameters))
    weight = node.get_parameter('weight_lateral_acceleration').value
    assert weight > 0.0
    np.testing.assert_allclose(objective - base, weight * np.mean((acceleration / node.ay_max)**2),
                              rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('coefficients', [(1.0, 0.4, -0.3, 0.2), (-2.0, 0.0, 0.0, 0.0)])
def test_dynamic_light_matches_unloaded_velocity_polynomial(coefficients, monkeypatch):
    from dataclasses import replace
    from plan_predictive_cost_map.planner.config import VehicleConfig
    from plan_predictive_cost_map import pcm_longitudinal_dynamics as models

    def forbidden(*args, **kwargs):
        raise AssertionError('dynamic_light constructed tire-slip dynamics')

    monkeypatch.setattr(models, 'dynamic_step', forbidden)
    monkeypatch.setattr(models, 'acceleration_command', forbidden)
    monkeypatch.setattr(models, 'acceleration_polynomial_step', forbidden)
    settings = replace(VehicleConfig(), air_density=0.0, rolling_resistance_coefficient=0.0,
                       viscous_resistance=0.0)
    rollout = models.build_dynamic_light_rollout(1.0, 4, 0.05, 0.25, settings)
    prediction = rollout(coefficients, ca.DM([10., 20., 3., 0., 8., 0., 0.]), 0., 0.,
                         ca.DM.zeros(4), ca.DM.zeros(4), 0., reference_data())
    times = np.arange(1, 5) * 0.25
    polynomial = np.asarray(coefficients)
    expected_speed = 8.0 + sum(polynomial[index] * times**(index + 1) for index in range(4))
    expected_progress = 8.0 * times + sum(polynomial[index] * times**(index + 2)
        / (index + 2) for index in range(4))
    np.testing.assert_allclose(prediction[1].full().flatten(), expected_speed, atol=1e-10)
    np.testing.assert_allclose(prediction[9].full().flatten(), expected_progress, atol=1e-7)
    np.testing.assert_allclose(prediction[0].full()[:, 0], expected_progress, atol=1e-7)
    np.testing.assert_allclose(prediction[8].full()[:, 0], 10.0 + expected_progress, atol=1e-7)
    expected_acceleration = sum((index + 1) * polynomial[index] * times**index for index in range(4))
    np.testing.assert_allclose(prediction[2].full()[:, 0], expected_acceleration)
    np.testing.assert_allclose(prediction[3].full().flatten(), expected_acceleration)
    np.testing.assert_allclose(float(prediction[4]), expected_speed[0], atol=1e-10)
    assert prediction[6].numel() == 41


@pytest.mark.parametrize('grade', [-0.1, 0.0, 0.1])
def test_dynamic_light_loads_match_high_accuracy_integration(grade):
    from scipy.integrate import solve_ivp
    from plan_predictive_cost_map.planner.config import VehicleConfig
    from plan_predictive_cost_map.pcm_longitudinal_dynamics import build_dynamic_light_rollout

    settings = VehicleConfig()
    def resistance(speed):
        return (0.5 * settings.air_density * settings.drag_coefficient * settings.frontal_area * speed**2
                / settings.mass + settings.viscous_resistance * speed / settings.mass
                + settings.rolling_resistance_coefficient * 9.81 * math.cos(grade) + 9.81 * math.sin(grade))

    def request(elapsed):
        speed = 8.0 + elapsed + 0.2 * elapsed**2 - 0.1 * elapsed**3 + 0.05 * elapsed**4
        acceleration = 1.0 + 0.4 * elapsed - 0.3 * elapsed**2 + 0.2 * elapsed**3
        return acceleration + resistance(speed)

    expected = solve_ivp(lambda elapsed, state: [state[1], request(elapsed) - resistance(state[1])],
                         (0.0, 1.0), [0.0, 8.0], t_eval=np.arange(1, 5) * 0.25, rtol=1e-12, atol=1e-12)
    rollout = build_dynamic_light_rollout(1.0, 4, 0.05, 0.25, settings,
                                         use_road_gradient=False, road_pitch=grade)
    prediction = rollout([1., 0.2, -0.1, 0.05], ca.DM([0., 0., 0., 0., 8., 0., 0.]), 0., 0.,
                         ca.DM.zeros(4), ca.DM.zeros(4), 0., reference_data())
    np.testing.assert_allclose(prediction[9].full().flatten(), expected.y[0], atol=1e-8)
    np.testing.assert_allclose(prediction[1].full().flatten(), expected.y[1], atol=1e-8)
    np.testing.assert_allclose(prediction[3].full().flatten(), request(np.arange(1, 5) * 0.25), atol=1e-10)


@pytest.mark.parametrize('local_grade,initial_pitch', [(0.1, 0.0), (0.0, 0.1), (-0.1, 0.0)])
def test_dynamic_light_uses_reference_grade_and_map_pitch(local_grade, initial_pitch):
    from dataclasses import replace
    from plan_predictive_cost_map.planner.config import VehicleConfig
    from plan_predictive_cost_map.pcm_longitudinal_dynamics import build_dynamic_light_rollout

    settings = replace(VehicleConfig(), air_density=0.0, rolling_resistance_coefficient=0.0,
                       viscous_resistance=0.0)
    points = np.array([[0., 0., 0.], [100.0 * math.cos(local_grade), 0., 100.0 * math.sin(local_grade)]])
    acceleration = -9.81 * math.sin(local_grade + initial_pitch)
    rollout = build_dynamic_light_rollout(1.0, 4, 0.05, 0.25, settings)
    prediction = rollout([acceleration, 0., 0., 0.], ca.DM([0., 0., 0., 0., 8., 0., 0.]), 0., 0.,
                         ca.DM.zeros(4), ca.DM.zeros(4), initial_pitch, reference_data(points))
    times = np.arange(1, 5) * 0.25
    progress = 8.0 * times + 0.5 * acceleration * times**2
    np.testing.assert_allclose(prediction[1].full().flatten(), 8.0 + acceleration * times, atol=1e-10)
    np.testing.assert_allclose(prediction[9].full().flatten(), progress, atol=1e-10)
    np.testing.assert_allclose(prediction[0].full()[:, 2], progress * math.sin(local_grade), atol=1e-10)
    np.testing.assert_allclose(prediction[8].full()[:, 2], progress * math.sin(local_grade + initial_pitch), atol=1e-10)
    np.testing.assert_allclose(prediction[3].full(), 0.0, atol=1e-10)


def test_dynamic_light_motion_is_independent_of_actuator_sampling():
    from plan_predictive_cost_map.pcm_longitudinal_dynamics import build_dynamic_light_rollout

    arc = np.linspace(0.0, 100.0, 301)
    points = np.column_stack((40.0 * np.sin(arc / 40.0), 40.0 * (1.0 - np.cos(arc / 40.0)),
                              0.001 * arc**2))
    predictions = []
    for step in (0.05, 0.1):
        rollout = build_dynamic_light_rollout(4.0, 20, step, 0.25)
        predictions.append(rollout([0.5, 0.1, 0.0, 0.0], ca.DM([0., 0., 0., 0., 8., 0., 0.]),
            0., 0., ca.DM.zeros(4), ca.DM.zeros(4), 0., reference_data(points)))
    for output in (0, 1, 9):
        np.testing.assert_allclose(predictions[0][output].full(), predictions[1][output].full(),
                                   atol=1e-12, rtol=0.0)


def test_dynamic_light_can_touch_zero_speed_and_reaccelerate():
    from dataclasses import replace
    from plan_predictive_cost_map.planner.config import VehicleConfig
    from plan_predictive_cost_map.pcm_longitudinal_dynamics import build_dynamic_light_rollout

    settings = replace(VehicleConfig(), air_density=0.0, rolling_resistance_coefficient=0.0,
                       viscous_resistance=0.0)
    rollout = build_dynamic_light_rollout(1.0, 4, 0.05, 0.25, settings)
    prediction = rollout([-1., 1., 0., 0.], ca.DM([0., 0., 0., 0., 0.25, 0., 0.]), 0., 0.,
                         ca.DM.zeros(4), ca.DM.zeros(4), 0., reference_data())
    times = np.arange(1, 5) * 0.25
    np.testing.assert_allclose(prediction[1].full().flatten(), (times - 0.5)**2, atol=1e-12)
    np.testing.assert_allclose(prediction[9].full().flatten(), 0.25 * times - 0.5 * times**2 + times**3 / 3.0, atol=1e-12)


@pytest.mark.parametrize('direction', [-1.0, 1.0])
def test_dynamic_actuator_bounds_use_unclipped_requests(node, direction):
    if node.prediction_model == 'kinematic':
        assert node.rollout_eval.size1_out(6) == 0
        return
    coefficients = np.zeros(node.M)
    coefficients[0] = direction * 20.0
    parameters = prediction_parameters(node)
    requested = node.rollout_eval(coefficients, parameters)[6].full().flatten()
    constraints = ca.Function('actuator_constraint_check', [node.c_sym, node.p_sym], [node.g])(
        coefficients, parameters).full().flatten()
    count = len(requested)
    np.testing.assert_allclose(constraints[7 * node.N:7 * node.N + count],
                               requested - node.vehicle_settings.maximum_acceleration)
    np.testing.assert_allclose(constraints[7 * node.N + count:7 * node.N + 2 * count],
                               -node.vehicle_settings.maximum_deceleration - requested)
    assert np.max(constraints[7 * node.N:7 * node.N + 2 * count]) > 0.0


def test_dynamic_light_reverse_trials_are_not_hidden_by_clipping():
    from plan_predictive_cost_map.pcm_longitudinal_dynamics import build_dynamic_light_rollout

    rollout = build_dynamic_light_rollout(2.0, 8, 0.05, 0.0)
    prediction = rollout([-5., 0., 0., 0.], ca.DM([0., 0., 0., 0., 1., 0., 0.]), 0., 0.,
                         ca.DM.zeros(4), ca.DM.zeros(4), 0., reference_data())
    speeds = prediction[1].full().flatten()
    progress = prediction[9].full().flatten()
    np.testing.assert_allclose(speeds, 1.0 - 5.0 * np.arange(1, 9) * 0.25 / 2.0)
    assert np.min(speeds) < 0.0
    assert np.min(progress) < 0.0
    assert float(prediction[4]) == 1.0


@pytest.mark.parametrize('node', ['dynamic_light'], indirect=True)
def test_dynamic_light_dense_speed_constraints_reject_intersample_reverse(node):
    parameters = prediction_parameters(node)
    coefficients = [-120., 400., 0., 0.]
    prediction = node.rollout_eval(coefficients, parameters)
    assert np.min(prediction[1].full()) > 0.0
    constraints = ca.Function('dense_speed_constraint_check', [node.c_sym, node.p_sym], [node.g])(
        coefficients, parameters).full().flatten()
    count = prediction[6].numel()
    offset = 7 * node.N + 2 * count
    assert np.max(constraints[offset:offset + count]) > 0.0


def test_dynamic_light_normalized_polynomial_and_selected_reference_channels():
    from plan_predictive_cost_map.pcm_lane_geometry import progress_reference_function
    from plan_predictive_cost_map.pcm_longitudinal_dynamics import velocity_polynomial

    times = ca.DM([0.0, 0.25, 1.0, 4.0])
    coefficients = ca.DM([1.0, -0.5, 0.2, 0.1])
    normalized = velocity_polynomial(times, coefficients, 8.0, 4.0)
    physical = velocity_polynomial(times, coefficients / ca.DM([4.0, 16.0, 64.0, 256.0]), 8.0)
    for actual, expected in zip(normalized, physical):
        np.testing.assert_allclose(actual.full(), expected.full(), atol=1e-12)
    reference = reference_data(np.array([[0., 0., 0.], [10., 0., 1.], [20., 10., 3.]]))
    full = progress_reference_function(96)
    selected = progress_reference_function(96, channels=(3, 4))
    progress = ca.MX.sym('selected_progress')
    full_values = full(progress, reference)[3:5]
    selected_values = selected(progress, reference)
    compare = ca.Function('selected_reference_derivatives', [progress],
        [selected_values - full_values, ca.jacobian(selected_values - full_values, progress),
         ca.jacobian(ca.jacobian(selected_values - full_values, progress), progress)])
    for distance in (0.0, 3.0, 10.0, 17.0):
        for difference in compare(distance):
            np.testing.assert_allclose(difference.full(), 0.0, atol=1e-12)


def test_fixed_basis_reference_matches_scipy_spline_and_derivative():
    from scipy.interpolate import make_interp_spline
    from plan_predictive_cost_map.pcm_lane_geometry import progress_reference_values

    arc = np.linspace(0.0, 80.0, 241)
    points = np.column_stack((40.0 * np.sin(arc / 40.0), 40.0 * (1.0 - np.cos(arc / 40.0)), 0.002 * arc**2))
    reference = reference_data(points)
    knots = make_interp_spline(np.linspace(0.0, 1.0, 96), np.zeros(96), k=3).t
    spline = make_interp_spline(np.linspace(0.0, 1.0, 96), np.zeros((96, 6)), k=3)
    spline.c = reference[2:].reshape(96, 6)
    assert np.array_equal(spline.t, knots)
    progress = ca.MX.sym('progress', 5)
    values = progress_reference_values(progress, reference, 96)
    evaluate = ca.Function('fixed_basis', [progress], [values, ca.jacobian(values[:, 0], progress)])
    distance = np.array([1.0, 12.5, 33.3, 61.0, 79.0])
    actual, slope = evaluate(distance)
    normalized = (distance - reference[0]) / (reference[1] - reference[0])
    np.testing.assert_allclose(actual.full(), spline(normalized), atol=1e-10)
    np.testing.assert_allclose(np.diag(slope.full()), spline.derivative()(normalized)[:, 0] / (reference[1] - reference[0]),
                               atol=1e-10)


def test_numeric_kernel_matches_casadi_kernels():
    from plan_predictive_cost_map.pcm_kernels import lane_kernel, numeric_rbf_kernel, rbf_kernel

    generator = np.random.default_rng(3)
    query, support = generator.normal(size=(7, 3)) * 20.0, generator.normal(size=(5, 3)) * 20.0
    np.testing.assert_allclose(numeric_rbf_kernel(query, support, [0.7, 2.0, 3.0, 1.5]),
                               rbf_kernel(ca.DM(query), ca.DM(support), [0.7, 2.0, 3.0, 1.5]).full(), atol=1e-14)
    np.testing.assert_allclose(numeric_rbf_kernel(query[:, :2], support[:, :2], [0.4, 3.0, 2.0]),
                               lane_kernel(ca.DM(query), ca.DM(support), [0.4, 3.0, 2.0]).full(), atol=1e-14)


def test_scenario_worker_process_matches_direct_parsing(node):
    import multiprocessing
    from rclpy.serialization import serialize_message
    from plan_predictive_cost_map.pcm_scenario import parse_scenario, scenario_worker

    scenario = Scenario()
    scenario.header.stamp.sec = 12
    path = PathWithTrafficRules()
    path.traffic_rules.maximum_speed = 9.0
    for distance in np.linspace(0.0, 60.0, 31):
        point = PathPointWithLaneId()
        point.point.pose.position.x = float(distance)
        point.point.pose.position.y = float(0.01 * distance**2)
        path.path.points.append(point)
    road = PathWithTrafficRules()
    for distance in np.linspace(0.0, 60.0, 31):
        point = PathPointWithLaneId()
        point.point.pose.position.x = float(distance)
        point.point.pose.position.y = 3.5
        road.path.points.append(point)
    scenario.paths = [path, road]
    context = multiprocessing.get_context('spawn')
    requests, results = context.Queue(), context.Queue()
    worker = context.Process(target=scenario_worker, args=(requests, results, node.scenario_settings), daemon=True)
    worker.start()
    try:
        requests.put(serialize_message(scenario))
        update = results.get(timeout=30.0)
    finally:
        requests.put(None)
        worker.join(timeout=10.0)
    assert not worker.is_alive()
    expected = parse_scenario(scenario, node.scenario_settings)
    assert update['stamp_ns'] == 12_000_000_000 and update['lane_received'] and update['target_speed'] == 9.0
    np.testing.assert_allclose(update['reference_data'], expected['reference_data'])
    for name in ('target_gp', 'drivable_gp'):
        for actual, reference in zip(update[name], expected[name]):
            np.testing.assert_allclose(actual, reference)
    node.apply_scenario(update)
    np.testing.assert_allclose(node.reference_data, expected['reference_data'])
    assert node.lane_received and node.target_speed == 9.0


def test_dynamic_light_time_shift_and_profile_initialization():
    from plan_predictive_cost_map.pcm_longitudinal_dynamics import (
        shift_velocity_coefficients, velocity_polynomial, velocity_profile_coefficients)

    coefficients = np.array([1.0, -2.0, 0.5, 0.3])
    delta, horizon = 0.37, 4.0
    times = ca.DM(np.linspace(0.0, 3.0, 7))
    shifted = shift_velocity_coefficients(coefficients, delta, horizon)
    expected = (velocity_polynomial(times + delta, coefficients, 6.0, horizon)[0]
                - velocity_polynomial(ca.DM(delta), coefficients, 6.0, horizon)[0] + 5.0)
    np.testing.assert_allclose(velocity_polynomial(times, shifted, 5.0, horizon)[0].full(), expected.full(), atol=1e-12)
    np.testing.assert_allclose(shift_velocity_coefficients(coefficients, 0.0, horizon), coefficients, atol=1e-15)
    profiles = velocity_profile_coefficients(horizon, 8.0, 13.0)
    assert len(profiles) == 10
    np.testing.assert_allclose(profiles[0], 0.0, atol=1e-12)
    braking = velocity_polynomial(ca.DM([1.0, 2.0]), profiles[3], 8.0, horizon)[0].full().flatten()
    np.testing.assert_allclose(braking, [6.0, 4.0], atol=0.3)


@pytest.mark.parametrize('node', ['dynamic_light'], indirect=True)
def test_dynamic_light_uses_shifted_previous_plan_only_when_nearly_feasible(node):
    node.reference_data = reference_data()
    node.p_numeric = prediction_parameters(node)
    node.ego_state.vx = 8.0
    node.target_speed = 8.0
    solver, initial, multipliers, initialization = node.light_initial_guess(10.0)
    assert solver is node.solver and initialization == 'profile' and multipliers == {}
    assert np.max(node.nlp_constraints(initial, node.p_numeric).full()) <= 0.0
    node.light_previous = np.array([-1.0, 0.2, 0.0, 0.0])
    node.light_previous_time = 9.95
    node.light_multipliers = {'lam_g0': np.zeros(node.g.numel())}
    solver, initial, multipliers, initialization = node.light_initial_guess(10.0)
    assert solver is node.light_warm_solver and initialization == 'shifted'
    np.testing.assert_allclose(initial, shift_velocity_coefficients(node.light_previous, 0.05, node.tau_h))
    node.light_previous = np.array([60.0, 0.0, 0.0, 0.0])
    solver, initial, multipliers, initialization = node.light_initial_guess(10.0)
    assert solver is node.solver and initialization == 'profile'
    assert np.max(node.nlp_constraints(initial, node.p_numeric).full()) <= 0.0


def test_dynamic_rollout_integrates_acceleration_and_braking():
    from dataclasses import replace
    from plan_predictive_cost_map.planner.config import VehicleConfig
    from plan_predictive_cost_map.pcm_longitudinal_dynamics import build_dynamic_rollout

    settings = replace(VehicleConfig(), air_density=0.0, rolling_resistance_coefficient=0.0,
                       viscous_resistance=0.0)
    rollout = build_dynamic_rollout(1.0, 4, 0.05, 0.25, settings)
    position, speed, acceleration, request, command, steering, dense_requests, orientation, states, progress = rollout(
        ca.DM.zeros(4), ca.DM([10., 20., 3., 0., 8., 0., 0.]), -2., 0.,
        ca.DM.zeros(4), ca.DM.zeros(4), 0., reference_data())
    times = np.arange(1, 5) * 0.25
    np.testing.assert_allclose(position.full()[:, 0], 8.0 * times - times**2, atol=1e-11)
    np.testing.assert_allclose(speed.full().flatten(), 8.0 - 2.0 * times, atol=1e-11)
    np.testing.assert_allclose(acceleration.full()[:, 0], -2.0, atol=1e-10)
    np.testing.assert_allclose(request.full().flatten(), -2.0, atol=1e-12)
    np.testing.assert_allclose(float(command), 7.5, atol=1e-12)
    np.testing.assert_allclose(steering.full(), 0.0, atol=1e-12)
    np.testing.assert_allclose(dense_requests.full(), -2.0, atol=1e-12)
    np.testing.assert_allclose(orientation.full(), np.tile([0., 0., 0., 1.], (4, 1)), atol=1e-12)
    np.testing.assert_allclose(states.full()[:, 4], speed.full().flatten(), atol=1e-12)


@pytest.mark.parametrize('coefficients', [(0., 0., 0.), (-2., 0., 0.), (-1., 0.3, -0.05)])
def test_kinematic_rollout_restores_velocity_polynomial(coefficients):
    from plan_predictive_cost_map.pcm_longitudinal_dynamics import build_kinematic_rollout

    rollout = build_kinematic_rollout(1.0, 4, 0.25)
    result = rollout(ca.DM(coefficients), ca.DM([0., 0., 0., 0., 8., 0., 0.]), 0., 0.,
                     ca.DM.zeros(4), ca.DM.zeros(4), 0., reference_data())
    times = np.arange(1, 5) * 0.25
    expected_speed = 8.0 + coefficients[0] * times + coefficients[1] * times**2 + coefficients[2] * times**3
    expected_distance = (8.0 * times + coefficients[0] * times**2 / 2.0
                         + coefficients[1] * times**3 / 3.0 + coefficients[2] * times**4 / 4.0)
    np.testing.assert_allclose(result[0].full()[:, 0], expected_distance, atol=1e-12)
    np.testing.assert_allclose(result[1].full().flatten(), expected_speed, atol=1e-12)
    np.testing.assert_allclose(float(result[4]), expected_speed[0], atol=1e-12)
    assert result[6].numel() == 0


def test_kinematic_mode_does_not_construct_dynamic_integrator(monkeypatch):
    from plan_predictive_cost_map import pcm_longitudinal

    def forbidden(*arguments, **keywords):
        raise AssertionError('Kinematic mode constructed the dynamic model')

    monkeypatch.setattr(pcm_longitudinal, 'build_dynamic_rollout', forbidden)
    rclpy.init(args=['--ros-args', '-p', 'prediction_model:=kinematic',
                     '-p', 'lookahead_time:=1.0', '-p', 'prediction_steps:=4'])
    planner = pcm_longitudinal.pcm_node()
    try:
        assert planner.M == 3
        assert planner.rollout_eval.size1_out(6) == 0
    finally:
        planner.destroy_node()
        rclpy.shutdown()


def test_dynamic_rollout_predicts_turning_pose():
    from plan_predictive_cost_map.pcm_longitudinal_dynamics import build_dynamic_rollout

    rollout = build_dynamic_rollout(1.0, 4, 0.05, 0.25)
    distances = np.linspace(0., 40., 81)
    reference = reference_data(np.column_stack((distances, 0.01 * distances**2, np.zeros(len(distances)))))
    prediction = rollout(ca.DM.zeros(4), ca.DM([0., 0., 0., 0., 8., 0., 0.]), 0., 0.,
                         ca.DM.zeros(4), ca.DM.zeros(4), 0., reference)
    assert float(prediction[0][-1, 1]) > 0.0
    assert float(prediction[8][-1, 6]) > 0.0
    assert float(prediction[7][-1, 2]) > 0.0
    np.testing.assert_allclose(np.linalg.norm(prediction[7].full(), axis=1), 1.0, atol=1e-12)


def test_vehicle_parameters_are_planner_owned(node):
    from dataclasses import fields
    from plan_predictive_cost_map.planner.config import VehicleConfig

    defaults = VehicleConfig()
    for setting in fields(defaults):
        assert node.get_parameter('vehicle.' + setting.name).value == getattr(defaults, setting.name)
    assert node.model_control_step == 0.05
    assert node.vehicle_settings.integration_step == 0.005


def test_dynamic_rollout_accounts_for_road_loads_and_stopped_state():
    from plan_predictive_cost_map.pcm_longitudinal_dynamics import build_dynamic_rollout

    rollout = build_dynamic_rollout(1.0, 4, 0.05, 0.25)
    initial = ca.DM([0., 0., 0., 0., 8., 0., 0.])
    level = rollout(ca.DM.zeros(4), initial, 0., 0., ca.DM.zeros(4), ca.DM.zeros(4), 0., reference_data())
    downhill = rollout(ca.DM.zeros(4), initial, 0., 0., ca.DM.zeros(4), ca.DM.zeros(4), -0.2, reference_data())
    np.testing.assert_allclose(level[1].full(), 8.0, atol=1e-10)
    np.testing.assert_allclose(downhill[1].full(), 8.0, atol=1e-10)
    assert float(downhill[3][0]) < float(level[3][0]) - 1.0
    stopped = rollout(ca.DM([-5., 0., 0., 0.]), ca.DM.zeros(7), 0., 0.,
                      ca.DM.zeros(4), ca.DM.zeros(4), 0., reference_data())
    np.testing.assert_allclose(stopped[1].full(), 0.0, atol=1e-12)
    np.testing.assert_allclose(stopped[0].full(), 0.0, atol=1e-12)


@pytest.mark.skipif(os.environ.get('PCM_DYNAMIC_MODEL_ROS_TEST') != '1',
                    reason='Opt-in C++ simulator dynamic-model parity test')
def test_acceleration_model_matches_live_cpp_simulator():
    from autoware_control_msgs.msg import Control
    from crp_msgs.msg import Ego
    from plan_predictive_cost_map.planner.dynamics import dynamic_step

    root = Path(__file__).resolve().parents[3]
    previous = {name: os.environ.get(name) for name in ('ROS_DOMAIN_ID', 'ROS_LOCALHOST_ONLY')}
    os.environ['ROS_DOMAIN_ID'] = str(160 + os.getpid() % 10)
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    rclpy.init(args=[])
    probe = rclpy.create_node('acceleration_model_cpp_parity')
    states = []
    subscription = probe.create_subscription(Ego, '/ego', states.append, 100)
    publisher = probe.create_publisher(Control, '/control/command/control_cmd', 10)
    process = subprocess.Popen([
        str(root / 'install/simulator/lib/simulator/simulator'), '--ros-args', '--log-level', 'error',
        '-p', f'scenario_file:={root / "src/scenario_files/occluded_pedestrian_costmap_data.mat"}',
        '-p', 'vehicle_model:=dynamic', '-p', 'command_mode:=acceleration',
        '-p', 'initial_x:=100.0', '-p', 'initial_y:=0.0', '-p', 'initial_yaw:=0.0', '-p', 'initial_speed:=10.0',
        '-p', 'simulation_time_stamps:=true', '-p', 'use_internal_controller:=false',
        '-p', 'use_road_gradient:=false', '-p', 'road_pitch:=0.05',
        '-p', 'enable_occlusion:=false', '-p', 'generate_virtual_objects:=false',
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        deadline = time.monotonic() + 15.0
        initial = None
        while time.monotonic() < deadline:
            rclpy.spin_once(probe, timeout_sec=0.02)
            if process.poll() is not None:
                raise RuntimeError(process.communicate()[1])
            if len(states) >= 3 and publisher.get_subscription_count() > 0:
                if initial is None:
                    initial = states[-1]
                command = Control()
                command.longitudinal.acceleration = 1.0
                command.lateral.steering_tire_angle = float(np.float32(0.04))
                publisher.publish(command)
            if states and states[-1].header.stamp.sec + states[-1].header.stamp.nanosec * 1e-9 >= 0.5:
                break
        assert initial is not None
        assert states[-1].header.stamp.sec + states[-1].header.stamp.nanosec * 1e-9 >= 0.5
        predicted = ca.DM([100., 0., 0., 0., 10., 0., 0.])
        step = dynamic_step(0.05)
        count = 0
        compared = 0
        for state in states:
            elapsed = state.header.stamp.sec + state.header.stamp.nanosec * 1e-9
            sample = round(elapsed / 0.05)
            if sample == 0:
                continue
            while count < sample:
                predicted = step(predicted, ca.DM([1.0, float(np.float32(0.04))]), ca.DM([0., 0.05]))[0]
                count += 1
            pose = state.pose.pose
            actual = [pose.position.x, pose.position.y, pose.position.z, float(state.orientation),
                      state.twist.twist.linear.x, state.twist.twist.linear.y, state.twist.twist.angular.z]
            np.testing.assert_allclose(predicted.full().flatten(), actual, atol=1e-6, rtol=1e-7)
            compared += 1
        assert compared >= 5
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        probe.destroy_node()
        rclpy.shutdown()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_optimizer_cannot_accelerate_above_scenario_speed(node):
    parameters = np.zeros(node.p_sym.numel())
    parameters[0] = 8.0
    parameters[-1] = 8.0
    evaluate = ca.Function('speed_limit_check', [node.c_sym, node.p_sym], [node.g])
    coefficients = np.zeros(node.M)
    coefficients[0] = 1.0
    values = evaluate(coefficients, parameters).full().flatten()
    assert np.all(values[3 * node.N:4 * node.N] > 0.0)
    parameters[0] = 10.0
    coefficients[0] = -1.0
    braking = evaluate(coefficients, parameters).full().flatten()
    assert np.all(braking[3 * node.N:4 * node.N] <= 0.0)


def test_publishes_scenario_elevation_and_control(node):
    scenario = Scenario()
    scenario.header.frame_id = 'base_link'
    path = PathWithTrafficRules()
    path.traffic_rules.maximum_speed = 8.0
    for distance in (0.0, 10.0, 10.0, 20.0, 30.0):
        point = PathPointWithLaneId()
        point.point.pose.position.x = distance
        point.point.pose.position.z = 2.0 + 0.1 * distance
        path.path.points.append(point)
    scenario.paths.append(path)
    node.scenario_callback(scenario)
    trajectories, commands = [], []
    node.trajectory_pub = SimpleNamespace(publish=trajectories.append)
    node.cmd_pub = SimpleNamespace(publish=commands.append)
    node.last_exec = 0.0
    node.callback()
    assert len(trajectories) == len(commands) == 1
    assert trajectories[0].header.frame_id == 'base_link'
    assert len(trajectories[0].points) == node.N
    for point in trajectories[0].points:
        np.testing.assert_allclose(point.pose.position.z, 2.0 + 0.1 * point.pose.position.x, atol=1e-6)
        assert point.longitudinal_velocity_mps >= 0.0
    assert commands[0].longitudinal.velocity >= 0.0
    assert node.target_speed == 8.0
    assert node.p_numeric[-1] == 8.0
    velocity_cost = ca.Function('velocity_cost', [node.c_sym, node.p_sym],
                                [node.get_cost_vel(node.c_sym, node.p_sym)])
    for speed in (4.0, 0.0):
        path.traffic_rules.maximum_speed = speed
        scenario.paths = [path]
        node.scenario_callback(scenario)
        node.last_exec = 0.0
        node.replan = True
        node.callback()
        assert node.target_speed == speed
        assert node.p_numeric[-1] == speed
        parameters = node.p_numeric.copy()
        parameters[0] = speed
        np.testing.assert_allclose(float(velocity_cost(np.zeros(node.M), parameters)), 0.0, atol=1e-12)
        parameters[0] = speed + 1.0
        assert float(velocity_cost(np.zeros(node.M), parameters)) > 0.0
    assert len(trajectories) == len(commands) == 3


def test_object_predictions_match_configured_grid(node):
    obj = ObjectState(x=10.0, y=2.0, theta=0.0, vx=2.0, ax=0.0, type=1,
                      predicted_path=np.array([[10.0, 2.0], [12.0, 3.0]]),
                      prediction_time_step=0.5)
    samples = node.sample_object_path(obj)
    np.testing.assert_allclose(samples, [[11.0, 2.5], [12.0, 3.0], [12.0, 3.0], [12.0, 3.0]])
    obj.predicted_path = np.empty((0, 2))
    np.testing.assert_allclose(node.sample_object_path(obj),
                               [[10.5, 2.0], [11.0, 2.0], [11.5, 2.0], [12.0, 2.0]])
    obj.predicted_path = np.array([[10.0, 2.0]])
    np.testing.assert_allclose(node.sample_object_path(obj), [[10.0, 2.0]] * node.N)


def test_gp_snapshot_reuses_solver_state(node, monkeypatch):
    publication_time = [10.0]
    monkeypatch.setattr('plan_predictive_cost_map.pcm_longitudinal.default_timer', lambda: publication_time[0])
    states = []
    node.publish_gp_state = True
    node.gp_state_pub = SimpleNamespace(publish=states.append, get_subscription_count=lambda: 1)
    scenario = Scenario()
    scenario.header.stamp.sec = 123
    scenario.header.stamp.nanosec = 456789
    path = PathWithTrafficRules()
    path.traffic_rules.maximum_speed = 8.0
    for distance in (0.0, 10.0, 20.0, 30.0):
        point = PathPointWithLaneId()
        point.point.pose.position.x = distance
        point.point.pose.position.z = 0.1 * distance
        path.path.points.append(point)
    scenario.paths = [path]
    node.scenario_callback(scenario)
    node.lanes.update(ego_lane_x=[0., 10., 20., 30.], ego_lane_y=[0.] * 4,
                      ego_lane_z=[0., 1., 2., 3.])
    node.p_numeric = prediction_parameters(node)
    evaluation = node.fsup_eval(np.zeros(node.M), node.p_numeric)
    node.publish_gp_snapshot(evaluation[1:])
    assert len(states) == 1
    viewer = CostmapViewer()
    try:
        assert viewer.settings['update_rate'] == 4.0
        assert viewer.timer.timer_period_ns == 250_000_000
        surfaces, clouds = [], []
        viewer.surface_pub = SimpleNamespace(publish=surfaces.append)
        viewer.cloud_pub = SimpleNamespace(publish=clouds.append)
        viewer.receive(SimpleNamespace(data='not-json'))
        assert viewer.snapshot is None
        viewer.receive(states[0])
        viewer.received_at = default_timer() - viewer.settings['stale_timeout'] - 1.0
        viewer.render()
        assert not surfaces
        viewer.receive(states[0])
        viewer.render()
        assert len(surfaces) == len(clouds) == 1
        assert len(surfaces[0].points) == 7800
        assert surfaces[0].lifetime.sec == 2
        assert surfaces[0].header.stamp == scenario.header.stamp
        assert clouds[0].header.stamp == scenario.header.stamp
        viewer.render()
        assert len(surfaces) == 1
    finally:
        viewer.destroy_node()
    snapshot = json.loads(states[0].data)
    assert snapshot['stamp_ns'] == 123_000_456_789
    validate_snapshot(snapshot)
    trajectory = node.traj_eval(np.zeros(node.M), node.p_numeric)
    queries = np.column_stack((trajectory[0].full().flatten(), trajectory[1].full().flatten(),
                               np.asarray(node.t).flatten()))
    np.testing.assert_allclose(evaluate_field(snapshot, queries, batch_size=2),
                               evaluation[0].full().flatten(), rtol=1e-9, atol=1e-9)
    lane_and_offroad = evaluate_field(snapshot, np.array([[4., 0., 0.5], [4., 0., 0.625],
                                                         [4., 10., 0.5]]))
    np.testing.assert_allclose(lane_and_offroad[0], lane_and_offroad[1], rtol=1e-12)
    assert lane_and_offroad[0] < lane_and_offroad[2] - 0.1
    expected_offroad = 0.5 * (-0.95 + np.sqrt(0.95**2 + 0.01))
    np.testing.assert_allclose(lane_and_offroad[2], expected_offroad, atol=1e-12)
    assert 0.0 < lane_and_offroad[2] < 0.003
    assert snapshot['version'] == 2
    for name in ('ego_lane', 'opposite_lane'):
        assert len(snapshot['layers'][name]['kernel']) == 3
        support_count = node.target_support_points if name == 'ego_lane' else node.drivable_support_points
        assert np.asarray(snapshot['layers'][name]['points']).shape == (support_count, 2)
        layer = snapshot['layers'][name]
        assert layer['offset'] == 0.0
        far_lane_mean = node.f_dyn_(ca.DM([[4., 10., 0.5]]), ca.DM(layer['points']), None,
                                    layer['kernel'], None, ca.DM(layer['alpha']))
        assert abs(float(far_lane_mean)) < 1e-18
    grid_x, grid_y = np.meshgrid(np.array([0., 2., 4.]), np.array([-1., 1.]))
    marker, cloud = surface_messages(snapshot, grid_x, grid_y, 0.5, 2.0, 0.15, 0.65, 2.0)
    dtype = np.dtype([(name, '<f4') for name in ('x', 'y', 'z', 'mean', 'road_z', 'prediction_time')]
                     + [('rgb', '<u4')])
    data = np.frombuffer(bytes(cloud.data), dtype=dtype)
    np.testing.assert_allclose(data['road_z'], grid_x.ravel() * 0.1, atol=1e-6)
    np.testing.assert_allclose(data['z'], data['road_z'] + 0.15 + 2.0 * data['mean'], atol=1e-6)
    assert len(marker.points) == 12
    assert len(marker.colors) == 12
    assert marker.header.frame_id == 'base_link'
    assert cloud.is_dense
    boundaries = [[0., -2., -0.4], [0., 2., 0.4], [30., -2., 2.6], [30., 2., 3.4]]
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    np.testing.assert_allclose(road_height(points, snapshot['road'], boundaries),
                               grid_x.ravel() * 0.1 + grid_y.ravel() * 0.2, atol=1e-9)
    assert snapshot['frame_id'] == 'base_link'
    assert len(snapshot['layers']['pedestrian']['points']) == node.pedestrian_support_count
    np.testing.assert_allclose(snapshot['layers']['ego_lane']['points'], evaluation[1].full())
    np.testing.assert_allclose(snapshot['layers']['ego_lane']['alpha'], evaluation[3].full().flatten())
    node.publish_gp_snapshot(evaluation[1:])
    assert len(states) == 1
    publication_time[0] = 10.0 + node.gp_state_period - 0.001
    node.publish_gp_snapshot(evaluation[1:])
    assert len(states) == 1
    publication_time[0] = 10.0 + node.gp_state_period
    node.publish_gp_snapshot(evaluation[1:])
    assert len(states) == 2


def test_viewer_shutdown_is_idempotent(monkeypatch):
    from rclpy.executors import ExternalShutdownException
    from plan_predictive_cost_map import pcm_costmap_viewer

    def shutdown_during_spin(viewer):
        rclpy.shutdown()
        raise ExternalShutdownException()

    monkeypatch.setattr(pcm_costmap_viewer.rclpy, 'spin', shutdown_during_spin)
    pcm_costmap_viewer.main(args=[])
    assert not rclpy.ok()


def test_lane_kernel_ignores_time_and_retains_xy_contrast(node):
    from plan_predictive_cost_map.pcm_kernels import lane_kernel

    queries = ca.DM([[4., 0., 0.5], [4., 0., 100.], [4., 10., 0.5]])
    support = ca.DM([[4., 0.]])
    for parameters in (node.p_el, node.p_ol):
        assert len(parameters) == 3
        actual = lane_kernel(queries, support, parameters).full().flatten()
        expected = parameters[0]**2 * np.exp(-0.5 * np.array([0., 0., (10. / parameters[2])**2]))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.)
        assert actual[0] > actual[2]


def test_target_gp_supports_and_coefficients_are_fixed_parameters(node, monkeypatch):
    scenario = Scenario()
    path = PathWithTrafficRules()
    path.traffic_rules.maximum_speed = 8.0
    for position_x, position_y in ((0., 0.), (10., 0.), (10., 10.), (0., 10.)):
        point = PathPointWithLaneId()
        point.point.pose.position.x, point.point.pose.position.y = position_x, position_y
        path.path.points.append(point)
    scenario.paths = [path]
    node.scenario_callback(scenario)
    def forbidden(*arguments, **keywords):
        raise AssertionError('Static GP fitting was called inside the optimizer expression')

    monkeypatch.setattr(ca, 'solve', forbidden)
    node.cost_sup(node.c_sym, node.p_sym)
    parameters = prediction_parameters(node)
    evaluation = node.fsup_eval(np.zeros(node.M), parameters)
    candidate = np.zeros(node.M)
    candidate[0] = -1.0
    changed = node.fsup_eval(candidate, parameters)
    np.testing.assert_allclose(evaluation[1].full(), node.target_gp_points)
    np.testing.assert_allclose(evaluation[3].full().flatten(), node.target_gp_alpha)
    np.testing.assert_allclose(changed[1].full(), evaluation[1].full())
    np.testing.assert_allclose(changed[3].full(), evaluation[3].full())
    assert node.model_rollout.n_in() == 8


def test_drivable_gp_uses_one_unordered_waypoint_set():
    from plan_predictive_cost_map.pcm_lane_geometry import fit_drivable_gp
    from plan_predictive_cost_map.pcm_kernels import lane_kernel

    polylines = [np.column_stack((np.linspace(0., 20., 21), np.full(21, lateral)))
                 for lateral in (0., 4., 8.)]
    kernel = [1., 5., 2.]
    waypoints = np.vstack(polylines)
    points, alpha = fit_drivable_gp(waypoints, 48, kernel, -0.4, 0.001)
    assert set(np.unique(points[:, 1])) >= {0., 4., 8.}
    shuffled_points, shuffled_alpha = fit_drivable_gp(waypoints[::-1], 48, kernel, -0.4, 0.001)
    np.testing.assert_allclose(points, shuffled_points)
    np.testing.assert_allclose(alpha, shuffled_alpha)
    queries = ca.DM([[10., 0.], [10., 4.], [10., 8.], [10., -15.]])
    means = (lane_kernel(queries, ca.DM(points), kernel) @ ca.DM(alpha)).full().flatten()
    assert np.all(means[:3] < -0.35)
    assert abs(means[-1]) < 1e-6


def test_scenario_road_paths_are_one_unordered_drivable_set(node):
    scenario = Scenario()
    target = PathWithTrafficRules()
    target.traffic_rules.maximum_speed = 8.0
    for distance in (0., 10., 20., 30.):
        point = PathPointWithLaneId()
        point.point.pose.position.x = distance
        target.path.points.append(point)
    roads = []
    for lateral in (0., 4., 8.):
        road = PathWithTrafficRules()
        for distance in (0., 10., 20., 30.):
            point = PathPointWithLaneId()
            point.point.pose.position.x = distance
            point.point.pose.position.y = lateral
            road.path.points.append(point)
        roads.append(road)
    scenario.paths = [target, *roads]
    node.scenario_callback(scenario)
    expected_points = node.drivable_points.copy()
    expected_alpha = node.drivable_alpha.copy()
    assert set(expected_points[:12, 1]) >= {0., 4., 8.}
    merged = PathWithTrafficRules()
    merged.path.points = [point for road in reversed(roads) for point in reversed(road.path.points)]
    scenario.paths = [target, merged]
    node.scenario_callback(scenario)
    np.testing.assert_allclose(node.drivable_points, expected_points)
    np.testing.assert_allclose(node.drivable_alpha, expected_alpha)


def test_rbf_kernel_is_stable_under_large_coordinate_translation():
    from plan_predictive_cost_map.pcm_kernels import rbf_kernel

    query = ca.MX.sym('translated_query', 1, 3)
    support = ca.DM([[1e12, 1e12, 0.5]])
    kernel = rbf_kernel(query, support, [1., 5., 2., 0.5])
    evaluate = ca.Function('translated_kernel', [query], [kernel, ca.jacobian(kernel, query)])
    value, derivative = evaluate(ca.DM([[1e12 + 1., 1e12 + 2., 0.5]]))
    np.testing.assert_allclose(float(value), np.exp(-0.5 * (0.2**2 + 1.0)), atol=1e-12)
    assert np.isfinite(derivative.full()).all()


def test_dynamic_kernel_retains_time_dependence(node):
    queries = ca.DM([[4., 0., 0.5], [4., 0., 1.5], [4., 2., 0.5]])
    support = ca.DM([[4., 0., 0.5]])
    for parameters in (node.p_ped, node.p_veh):
        assert len(parameters) == 4
        differences = (queries.full() - support.full()) / np.asarray(parameters[1:])
        expected = parameters[0]**2 * np.exp(-0.5 * np.sum(differences**2, axis=1))
        np.testing.assert_allclose(node.k(queries, support, parameters).full().flatten(), expected,
                                   rtol=1e-12, atol=1e-12)


def test_virtual_pedestrian_probabilities_and_branches_are_retained(node):
    from autoware_perception_msgs.msg import PredictedObject, ObjectClassification, PredictedPath
    from geometry_msgs.msg import Pose

    scenario = Scenario()
    pedestrian = PredictedObject()
    pedestrian.existence_probability = 0.2
    pedestrian.classification = [ObjectClassification(label=7, probability=1.0)]
    pedestrian.kinematics.initial_pose_with_covariance.pose.position.x = 10.0
    pedestrian.kinematics.initial_pose_with_covariance.pose.orientation.w = 1.0
    for confidence, speed in ((0.9, 0.8), (0.1, 3.0)):
        branch = PredictedPath()
        branch.confidence = confidence
        branch.time_step.nanosec = 500_000_000
        for elapsed in (0.0, 0.5, 1.0):
            pose = Pose()
            pose.position.x = 10.0
            pose.position.y = speed * elapsed
            branch.path.append(pose)
        pedestrian.kinematics.predicted_paths.append(branch)
    scenario.local_moving_objects.objects = [pedestrian]
    scenario.virtual_object_ids = [pedestrian.object_id]
    node.scenario_callback(scenario)
    assert len(node.pedestrians) == 1
    parsed = node.pedestrians[0]
    np.testing.assert_allclose(parsed.existence_probability, 0.2)
    np.testing.assert_allclose([branch.confidence for branch in parsed.predicted_branches], [0.9, 0.1])
    assert len(parsed.predicted_branches) == 2
    points, alpha, metadata = node.fit_dynamic_gp(
        [parsed], node.max_pedestrians, node.p_ped, node.pedestrian_cost_primitive,
        node.pedestrian_noise_base, node.pedestrian_noise_growth)
    times = np.asarray(node.t).flatten()
    np.testing.assert_allclose(points[:node.N, 1], 0.8 * times)
    np.testing.assert_allclose(points[node.N:2 * node.N, 1], 3.0 * times)
    np.testing.assert_allclose(points[:node.N, 2], times)
    for branch_index, branch in enumerate(parsed.predicted_branches):
        section = slice(branch_index * node.N, (branch_index + 1) * node.N)
        probability = parsed.existence_probability * branch.confidence
        np.testing.assert_allclose(metadata['targets'][section], 1.0 + probability * 3.0)
        expected_noise = (node.pedestrian_noise_base + node.pedestrian_noise_growth * times
                          + probability * (1.0 - probability) * 3.0**2)
        np.testing.assert_allclose(metadata['noise_variances'][section], expected_noise)
        np.testing.assert_allclose(np.diff(metadata['noise_variances'][section]),
                                   node.pedestrian_noise_growth * node.dt)
        np.testing.assert_allclose(metadata['trajectory_confidence'][section], branch.confidence)
    np.testing.assert_allclose(metadata['targets'][2 * node.N:], 0.0)
    np.testing.assert_allclose(alpha[2 * node.N:], 0.0, atol=1e-12)
    matrix = node.k(ca.DM(points), ca.DM(points), node.p_ped).full()
    np.testing.assert_allclose((matrix + np.diag(metadata['noise_variances'])) @ alpha,
                               metadata['targets'], atol=1e-10)
    path = PathWithTrafficRules()
    path.traffic_rules.maximum_speed = 8.0
    for distance in (0.0, 10.0, 20.0, 30.0):
        point = PathPointWithLaneId()
        point.point.pose.position.x = distance
        path.path.points.append(point)
    scenario.paths = [path]
    node.scenario_callback(scenario)
    states = []
    node.publish_gp_state = True
    node.gp_state_pub = SimpleNamespace(publish=states.append, get_subscription_count=lambda: 1)
    node.last_exec = 0.0
    node.callback()
    assert len(states) == 1
    snapshot = json.loads(states[0].data)
    validate_snapshot(snapshot)
    np.testing.assert_allclose(snapshot['layers']['pedestrian']['targets'], metadata['targets'])
    np.testing.assert_allclose(snapshot['layers']['pedestrian']['noise_variances'], metadata['noise_variances'])
    trajectory = node.traj_eval(node.c_numeric, node.p_numeric)
    query = np.column_stack((trajectory[0].full().flatten(), trajectory[1].full().flatten(), times))
    np.testing.assert_allclose(evaluate_field(snapshot, query),
                               node.fsup_eval(node.c_numeric, node.p_numeric)[0].full().flatten(),
                               rtol=1e-8, atol=1e-8)
    pedestrian.existence_probability = 0.0
    scenario.local_moving_objects.objects = [pedestrian]
    node.scenario_callback(scenario)
    assert not node.pedestrians


@pytest.mark.parametrize('kernel,severity,noise_base,noise_growth', [
    ([1.0, 5.0, 1.5, 0.5], 3.0, 0.02, 0.25),
    ([math.exp(-1.0), math.exp(1.0), 2.2, 1.0], 1.0, 0.01, 0.05),
])
def test_unlikely_trajectory_cost_fades_without_suppressing_likely_trajectory(
        kernel, severity, noise_base, noise_growth):
    from plan_predictive_cost_map.pcm_kernels import rbf_kernel
    from plan_predictive_cost_map.pcm_longitudinal import PredictionBranch

    times = np.linspace(0.2, 4.0, 20)
    sample_times = np.linspace(0.0, 4.0, 41)
    helper = SimpleNamespace(N=20, max_predicted_paths=2, t=ca.DM(times), dt=0.2, k=rbf_kernel)
    helper.sample_object_path = lambda obj, branch: pcm_node.sample_object_path(helper, obj, branch)
    queries = ca.DM([[10.0, -2.0 + speed * 4.0, 4.0] for speed in (0.8, 3.0)])
    slow_means = []
    fast_means = []
    for slow_confidence in (0.5, 0.1, 0.01, 0.001):
        branches = [PredictionBranch(
            np.column_stack((np.full(41, 10.0), -2.0 + speed * sample_times)), 0.1, confidence)
            for speed, confidence in ((0.8, slow_confidence), (3.0, 1.0 - slow_confidence))]
        obj = ObjectState(10.0, -2.0, 0.0, 0.0, 0.0, 7, existence_probability=1.0,
                          predicted_branches=branches)
        points, alpha, metadata = pcm_node.fit_dynamic_gp(
            helper, [obj], 1, kernel, severity, noise_base, noise_growth)
        confidence = np.asarray(metadata['trajectory_confidence'])
        np.testing.assert_allclose(metadata['targets'], 1.0 + severity * confidence)
        np.testing.assert_allclose(metadata['noise_variances'],
            noise_base + noise_growth * np.tile(times, 2) + confidence * (1.0 - confidence) * severity**2)
        means = (rbf_kernel(queries, ca.DM(points), kernel) @ ca.DM(alpha)).full().flatten() - 1.0
        slow_means.append(means[0])
        fast_means.append(means[1])
    assert np.all(np.diff(slow_means) < 0.0)
    assert np.all(np.diff(fast_means) > 0.0)
    assert slow_means[-1] < 0.05
    assert fast_means[-1] > 0.05


def test_adjacent_oncoming_vehicle_does_not_make_waiting_infeasible():
    from plan_predictive_cost_map.pcm_kernels import rbf_kernel

    launch_file = Path(__file__).resolve().parents[1] / 'launch' / 'pcm_longitudinal.launch.py'
    description = runpy.run_path(str(launch_file))['generate_launch_description']()
    context = LaunchContext()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    planner = next(action for action in description.entities if isinstance(action, Node))
    normalized = next(value for key, value in vars(planner).items() if key.endswith('__parameters'))
    settings = evaluate_parameters(context, normalized)[0]
    count = settings['prediction_steps']
    times = np.linspace(settings['lookahead_time'] / count, settings['lookahead_time'], count)
    points = ca.DM(np.column_stack((20.0 - 10.0 * times, np.full(count, 3.4), times)))
    kernel = settings['vehicle_kernel']
    noise = settings['vehicle_noise_base'] + settings['vehicle_noise_growth'] * times
    alpha = np.linalg.solve(rbf_kernel(points, points, kernel).full() + np.diag(noise),
                             np.full(count, 1.0 + settings['vehicle_cost']))
    waiting = ca.DM(np.column_stack((np.zeros(count), np.zeros(count), times)))
    in_vehicle_lane = ca.DM(np.column_stack((np.zeros(count), np.full(count, 3.4), times)))
    waiting_cost = (rbf_kernel(waiting, points, kernel) @ ca.DM(alpha)).full().flatten() - 1.0
    conflict_cost = (rbf_kernel(in_vehicle_lane, points, kernel) @ ca.DM(alpha)).full().flatten() - 1.0
    assert np.max(waiting_cost) < settings['max_gp_cost']
    assert np.max(conflict_cost) > settings['max_gp_cost']


def test_existence_probability_changes_dynamic_cost(node):
    from dataclasses import replace

    certain = ObjectState(10., 0., 0., 0., 0., 7,
                          predicted_path=np.tile([10., 0.], (3, 1)), prediction_time_step=0.5)
    uncertain = replace(certain, existence_probability=0.2)
    query = ca.DM([[10., 0., 0.5]])
    means = []
    for pedestrian in (certain, uncertain):
        points, alpha, metadata = node.fit_dynamic_gp(
            [pedestrian], node.max_pedestrians, node.p_ped, node.pedestrian_cost_primitive,
            node.pedestrian_noise_base, node.pedestrian_noise_growth)
        means.append(float(node.k(query, ca.DM(points), node.p_ped) @ ca.DM(alpha)) - 1.0)
    assert means[0] > means[1]
    for objects in ([], [replace(certain, existence_probability=0.0)],
                    [replace(certain, trajectory_confidence=0.0)]):
        points, alpha, metadata = node.fit_dynamic_gp(
            objects, node.max_pedestrians, node.p_ped, node.pedestrian_cost_primitive,
            node.pedestrian_noise_base, node.pedestrian_noise_growth)
        np.testing.assert_allclose(alpha, 0.0)
        np.testing.assert_allclose(metadata['targets'], 0.0)


def test_optimizer_trajectory_and_command_use_selected_rollout(node):
    parameters = np.zeros(node.p_sym.numel())
    parameters[0] = 8.0
    parameters[-1] = 8.0
    coefficients = np.zeros(node.M)
    coefficients[:2] = [-1.0, 0.2]
    prediction = node.rollout_eval(coefficients, parameters)
    trajectory = node.traj_eval(coefficients, parameters)
    np.testing.assert_allclose(trajectory[0].full().flatten(), prediction[0].full()[:, 0])
    np.testing.assert_allclose(trajectory[1].full().flatten(), prediction[0].full()[:, 1])
    np.testing.assert_allclose(trajectory[2].full(), prediction[1].full())
    np.testing.assert_allclose(float(node.command_eval(coefficients, parameters)), float(prediction[4]))
    assert float(prediction[4]) < 8.0
    if node.prediction_model != 'kinematic':
        assert prediction[6].numel() > node.N
    else:
        assert prediction[6].numel() == 0


def test_vehicle_mass_changes_dynamic_response():
    from dataclasses import replace
    from plan_predictive_cost_map.planner.config import VehicleConfig
    from plan_predictive_cost_map.pcm_longitudinal_dynamics import acceleration_polynomial_step

    state = ca.DM([0., 0., 0., 0., 20., 0., 0.])
    settings = VehicleConfig()
    light = acceleration_polynomial_step(0.05, settings)(state, ca.DM.zeros(4), 0., 0., ca.DM.zeros(2))[0]
    heavy = acceleration_polynomial_step(0.05, replace(settings, mass=3000.0))(
        state, ca.DM.zeros(4), 0., 0., ca.DM.zeros(2))[0]
    assert float(heavy[4]) > float(light[4])


def test_brakes_for_an_occupied_crossing(node):
    scenario = Scenario()
    path = PathWithTrafficRules()
    path.traffic_rules.maximum_speed = 8.0
    for distance in (0.0, 10.0, 20.0, 30.0):
        point = PathPointWithLaneId()
        point.point.pose.position.x = distance
        path.path.points.append(point)
    scenario.paths = [path]
    node.scenario_callback(scenario)
    node.ego_state.vx = 8.0
    node.pedestrians = [ObjectState(13.5, 0., 0., 0., 0., 7,
                        predicted_path=np.tile([13.5, 0.], (3, 1)), prediction_time_step=0.5)]
    commands = []
    node.cmd_pub = SimpleNamespace(publish=commands.append)
    node.last_exec = 0.0
    node.callback()
    assert len(commands) == 1
    assert commands[0].longitudinal.velocity < 7.5


def test_invalid_gp_snapshot_does_not_crash_or_publish(node):
    scenario = Scenario()
    node.scenario_callback(scenario)
    node.p_numeric = np.zeros(node.p_sym.numel())
    node.p_numeric[0] = 8.0
    node.p_numeric[-1] = 8.0
    lane_state = list(node.fsup_eval(np.zeros(node.M), node.p_numeric)[1:])
    lane_state[0][0, 0] = np.nan
    snapshots = []
    node.publish_gp_state = True
    node.gp_state_pub = SimpleNamespace(publish=snapshots.append, get_subscription_count=lambda: 1)
    node.publish_gp_snapshot(lane_state)
    assert not snapshots


@pytest.mark.parametrize('successful_but_unsafe,nonfinite', [(False, False), (True, False), (False, True)])
def test_invalid_solver_output_does_not_publish_a_replacement(node, successful_but_unsafe, nonfinite):
    scenario = Scenario()
    path = PathWithTrafficRules()
    path.traffic_rules.maximum_speed = 8.0
    for distance in (0.0, 10.0, 20.0, 30.0):
        point = PathPointWithLaneId()
        point.point.pose.position.x = distance
        path.path.points.append(point)
    scenario.paths = [path]
    node.scenario_callback(scenario)
    node.ego_state.vx = 8.0
    commands, trajectories = [], []
    node.cmd_pub = SimpleNamespace(publish=commands.append)
    node.trajectory_pub = SimpleNamespace(publish=trajectories.append)
    snapshots = []
    node.publish_gp_state = True
    node.gp_state_pub = SimpleNamespace(publish=snapshots.append, get_subscription_count=lambda: 1)

    def rejected_solution(**arguments):
        coefficients = np.zeros(node.M)
        coefficients[0] = 5.0
        constraints = ca.DM(np.full(node.g.numel(), np.nan)) if nonfinite else ca.DM.ones(node.g.numel())
        return {'x': ca.DM(coefficients), 'g': constraints, 'f': ca.DM(1.0)}

    rejected_solution.stats = lambda: {'success': successful_but_unsafe, 'return_status': 'Rejected'}
    node.solver = rejected_solution
    node.last_exec = 0.0
    node.callback()
    assert not commands
    assert not trajectories
    assert node.replan
    assert len(snapshots) == 1
    diagnostics = json.loads(snapshots[0].data)['planning']
    assert not diagnostics['accepted']
    if nonfinite:
        assert diagnostics['maximum_constraint'] is None


def test_safety_cost_matches_matlab_mean_squared_field(node):
    coefficients = np.zeros(node.M)
    parameters = np.zeros(node.p_sym.numel())
    parameters[0] = 8.0
    parameters[-1] = 8.0
    safety, field = node.cost_sup(node.c_sym, node.p_sym)
    evaluate = ca.Function('matlab_safety_check', [node.c_sym, node.p_sym], [safety, field])
    actual, values = evaluate(coefficients, parameters)
    shifted = values.full().flatten() + 1.0
    np.testing.assert_allclose(float(actual), np.mean(shifted**2), rtol=1e-12)
    assert not np.isclose(float(actual), np.sum(shifted)**2 / node.N)


def test_collision_penalty_is_part_of_objective(node):
    coefficients = np.zeros(node.M)
    parameters = np.zeros(node.p_sym.numel())
    parameters[0] = 8.0
    parameters[-1] = 8.0
    field = node.fsup_eval(coefficients, parameters)[0]
    safety_function = ca.Function('safety_check', [node.c_sym, node.p_sym],
                                   [node.cost_sup(node.c_sym, node.p_sym)[0]])
    safety = float(safety_function(coefficients, parameters))
    velocity = float(node.get_cost_vel(coefficients, parameters))
    crash = float(ca.sum1(ca.exp(5.0 * node.softplus(field)) - 1.0))
    baseline = (node.get_parameter('weight_safety').value * safety
                + node.get_parameter('weight_velocity').value * velocity)
    actual = float(node.solver.get_function('nlp_f')(x=coefficients, p=parameters)['f'])
    np.testing.assert_allclose(actual, baseline + node.get_parameter('weight_crash').value * crash,
                               rtol=1e-9, atol=1e-9)
    assert actual > baseline


def run_crossing_case(overrides=None, simulation_duration=14.0, production_launch=False):
    from scipy.io import loadmat
    from autoware_control_msgs.msg import Control
    from crp_msgs.msg import Ego
    from std_msgs.msg import String
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    root = Path(__file__).resolve().parents[3]
    package = root / 'src' / 'PredictiveCostMapGenerator'
    description = runpy.run_path(str(package / 'launch' / 'pcm_longitudinal.launch.py'))['generate_launch_description']()
    context = LaunchContext()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    action = next(action for action in description.entities if isinstance(action, Node))
    normalized = next(value for key, value in vars(action).items() if key.endswith('__parameters'))
    parameters = evaluate_parameters(context, normalized)[0]
    parameters.update(publish_gp_state=True, gp_state_period=0.5)
    parameters.update(overrides or {})
    arguments = [str(root / 'install/plan_predictive_cost_map/lib/plan_predictive_cost_map/pcm_longitudinal'),
                 '--ros-args', '--log-level', 'error']
    for name, value in parameters.items():
        arguments.extend(['-p', f'{name}:={json.dumps(value)}'])
    if production_launch:
        if overrides:
            raise ValueError('Production-launch regression uses the saved launch defaults')
        arguments = ['ros2', 'launch', 'plan_predictive_cost_map', 'pcm_longitudinal.launch.py']
    previous = {name: os.environ.get(name) for name in ('ROS_DOMAIN_ID', 'ROS_LOCALHOST_ONLY')}
    os.environ['ROS_DOMAIN_ID'] = str(140 + os.getpid() % 20)
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    child_environment = os.environ.copy()
    if production_launch:
        for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
            child_environment.pop(name, None)
    rclpy.init(args=[])
    probe = rclpy.create_node('longitudinal_crossing_probe')
    egos, scenarios, commands, snapshots, source_lags = [], [], [], [], []

    def receive_snapshot(message):
        snapshots.append(message)
        if egos:
            stamp = egos[-1].header.stamp
            latest = stamp.sec * 1_000_000_000 + stamp.nanosec
            source_lags.append(max(0.0, (latest - json.loads(message.data)['stamp_ns']) * 1e-9))

    subscriptions = [
        probe.create_subscription(Ego, '/ego', egos.append, 100),
        probe.create_subscription(Scenario, '/scenario', scenarios.append, 1),
        probe.create_subscription(Control, '/control/command/control_cmd', commands.append, 100),
        probe.create_subscription(String, '/plan/pcm_longitudinal/gp_state', receive_snapshot,
                                  QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)),
    ]
    processes = []
    scenario_file = root / 'src/scenario_files/occluded_pedestrian_costmap_data.mat'
    try:
        processes.append(subprocess.Popen(arguments, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           text=True, env=child_environment, start_new_session=True))
        deadline = time.monotonic() + 10.0
        while (not any(endpoint.node_name == 'pcm_longitudinal'
                   for endpoint in probe.get_subscriptions_info_by_topic('/scenario'))
               and time.monotonic() < deadline):
            rclpy.spin_once(probe, timeout_sec=0.05)
        simulator_arguments = [
            str(root / 'install/simulator/lib/simulator/simulator'), '--ros-args', '--log-level', 'error',
            '-p', f'scenario_file:={scenario_file}', '-p', 'command_mode:=speed',
            '-p', 'use_internal_controller:=true', '-p', 'simulation_time_stamps:=true',
            '-p', 'initial_speed:=13.88888888888889', '-p', 'generate_virtual_objects:=true',
        ]
        if production_launch:
            simulator_arguments = ['ros2', 'launch', 'simulator', 'simulator.launch.py',
                                   'simulation_time_stamps:=true']
        processes.append(subprocess.Popen(simulator_arguments, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, text=True, env=child_environment,
                                           start_new_session=True))
        deadline = time.monotonic() + simulation_duration * 4.0 + 15.0
        while time.monotonic() < deadline:
            rclpy.spin_once(probe, timeout_sec=0.05)
            if any(process.poll() is not None for process in processes):
                errors = [process.communicate()[1] for process in processes if process.poll() is not None]
                raise RuntimeError('Planner or simulator exited during the crossing run: ' + '\n'.join(errors))
            if egos:
                elapsed = egos[-1].header.stamp.sec + egos[-1].header.stamp.nanosec * 1e-9
                if elapsed >= simulation_duration:
                    break
        assert egos and commands and snapshots, 'Crossing run produced no planner output'
        assert elapsed >= simulation_duration, 'Simulator did not reach the requested duration'
        data = loadmat(scenario_file, simplify_cells=True)
        pedestrian = data['pedestrian']['position']
        minimum_clearance = math.inf
        approach_speeds = []
        crossing_speed = None
        for ego in egos:
            elapsed = ego.header.stamp.sec + ego.header.stamp.nanosec * 1e-9
            pose = ego.pose.pose
            pedestrian_x = np.interp(elapsed, data['time'], pedestrian['x'])
            pedestrian_y = np.interp(elapsed, data['time'], pedestrian['y'])
            yaw = math.atan2(2.0 * (pose.orientation.w * pose.orientation.z + pose.orientation.x * pose.orientation.y),
                             1.0 - 2.0 * (pose.orientation.y**2 + pose.orientation.z**2))
            delta_x, delta_y = pedestrian_x - pose.position.x, pedestrian_y - pose.position.y
            longitudinal = math.cos(yaw) * delta_x + math.sin(yaw) * delta_y
            lateral = -math.sin(yaw) * delta_x + math.cos(yaw) * delta_y
            clearance = math.hypot(max(abs(longitudinal) - 2.25, 0.0),
                                    max(abs(lateral) - 0.925, 0.0)) - 0.3
            minimum_clearance = min(minimum_clearance, clearance)
            if 70.0 <= pose.position.x <= 103.0:
                approach_speeds.append(ego.twist.twist.linear.x)
            if pose.position.x >= 97.0 and crossing_speed is None:
                crossing_speed = ego.twist.twist.linear.x
        maximum_pedestrians = 0
        near_crossing_inputs = []
        for scenario in scenarios:
            pedestrians = [actor for actor in scenario.local_moving_objects.objects
                           if actor.classification and actor.classification[0].label == 7]
            maximum_pedestrians = max(maximum_pedestrians, len(pedestrians))
            for actor in pedestrians:
                position = actor.kinematics.initial_pose_with_covariance.pose.position
                if 0.0 < position.x < 30.0 and actor.existence_probability > 0.1:
                    near_crossing_inputs.append((round(position.x, 1), round(position.y, 1),
                                                 round(actor.existence_probability, 2)))
        peak_cost = -math.inf
        solver_outcomes = {}
        for message in snapshots:
            snapshot = json.loads(message.data)
            planning = snapshot.get('planning', {})
            status = planning.get('status', 'unknown')
            solver_outcomes[status] = solver_outcomes.get(status, 0) + 1
            query = np.column_stack((snapshot['layers']['ego_lane']['points'], snapshot['times']))
            peak_cost = max(peak_cost, float(np.max(evaluate_field(snapshot, query))))
        command_times = np.array([command.stamp.sec + command.stamp.nanosec * 1e-9 for command in commands])
        return {'minimum_clearance': minimum_clearance,
                'minimum_approach_speed': min(approach_speeds) if approach_speeds else None,
                'crossing_speed': crossing_speed, 'final_x': egos[-1].pose.pose.position.x,
                'maximum_pedestrians': maximum_pedestrians, 'maximum_planned_cost': peak_cost,
                'command_count': len(commands), 'ego_count': len(egos),
                'maximum_source_lag_sim_seconds': max(source_lags, default=0.0),
                'mean_source_lag_sim_seconds': float(np.mean(source_lags)) if source_lags else 0.0,
                'solver_outcomes': solver_outcomes,
                'maximum_command_gap_seconds': float(np.max(np.diff(command_times))) if len(command_times) > 1 else math.inf,
                'near_crossing_inputs': near_crossing_inputs[::max(1, len(near_crossing_inputs) // 10)]}
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        probe.destroy_node()
        rclpy.shutdown()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.mark.skipif(os.environ.get('PCM_LONGITUDINAL_CROSSING_TEST') != '1',
                    reason='Opt-in closed-loop simulator regression')
def test_default_crossing_yields_and_resumes():
    result = run_crossing_case(simulation_duration=22.0, production_launch=True)
    print('Crossing metrics:', json.dumps(result, sort_keys=True))
    assert result['minimum_clearance'] > 0.5
    assert result['minimum_approach_speed'] < 1.0
    assert result['crossing_speed'] is not None
    assert result['final_x'] > 110.0
    assert result['command_count'] > 150
    assert result['maximum_command_gap_seconds'] < 1.0
    assert all(status in ('Solve_Succeeded', 'Solved_To_Acceptable_Level')
               for status in result['solver_outcomes'])