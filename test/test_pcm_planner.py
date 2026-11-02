"""Focused mathematical and runtime regression tests for the new planner."""

from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from plan_predictive_cost_map.planner.config import PlannerConfig, VehicleConfig
from plan_predictive_cost_map.planner.dynamics import acceleration_command, dynamic_step
from plan_predictive_cost_map.planner.costmap import Branch, fit_costmap, fit_posterior, kernel, symbolic_cost
from plan_predictive_cost_map.planner.geometry import footprints_overlap, longitudinal_tracking_index, new_virtual_hazard
from plan_predictive_cost_map.planner.geometry import Road
from plan_predictive_cost_map.planner.optimizer import PlanningFailure, PolynomialMPC, command_seed, internal_steering, objective_terms
from shapely.geometry import box


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('g++ is needed for simulator model parity checks')
    test_directory = Path(__file__).resolve().parent
    simulator = test_directory.parent.parent / 'CooperativeResearchPlatform_Simulator'
    executable = tmp_path_factory.mktemp('model_oracle') / 'oracle'
    subprocess.run([compiler, '-std=c++17', '-O2', '-I' + str(simulator / 'include'),
                    str(test_directory / 'model_oracle.cpp'),
                    str(simulator / 'src/vehicle_model.cpp'), '-o', str(executable)], check=True)
    return executable


@pytest.mark.parametrize('speed', [0.0, 0.2, 1.0, 2.1, 15.0])
@pytest.mark.parametrize('acceleration,steering,pitch', [(2.0, 0.1, 0.0), (-4.0, -0.15, 0.1)])
def test_dynamic_model_matches_simulator(oracle, speed, acceleration, steering, pitch):
    initial = np.array([1.0, 2.0, 3.0, 0.2, speed, 0.03, 0.02])
    result = subprocess.check_output([str(oracle), *map(str, initial), str(acceleration),
                                      str(steering), str(pitch), str(pitch)], text=True)
    expected = np.fromstring(result, sep=' ')
    state, accelerations = dynamic_step(0.05)(initial, [acceleration, steering], [pitch, pitch])
    np.testing.assert_allclose(np.concatenate([np.array(state).ravel(), np.array(accelerations).ravel()]),
                               expected, atol=1e-10, rtol=1e-10)


def test_configuration_validation():
    assert PlannerConfig().steps == 60
    with pytest.raises(ValueError):
        VehicleConfig(front_axle_distance=3.0)
    with pytest.raises(ValueError):
        PlannerConfig(control_step=0.07)


def test_gp_matches_matlab_linear_solve():
    points = np.array([[0., 0., 0.], [1., 2., 0.5], [3., 0., 1.]])
    targets, noise, scales = np.array([2., 3., 1.]), np.array([.1, .3, .2]), np.array([2., 1., 1.])
    expected = np.linalg.solve(kernel(points, points, scales) + np.diag(noise) + np.eye(3) * 1e-8, targets)
    np.testing.assert_allclose(fit_posterior(points, targets, noise, scales), expected)


def test_symbolic_gp_and_probability_semantics():
    config = PlannerConfig(gp_support_points=8)
    route = np.array([[0., 0., 0.], [30., 0., 0.]])
    branch = Branch(np.array([[5., 0., 0.], [8., 0., 0.]]), np.array([0., 3.]), .7, .2, 0, (4., 2., 1.5))
    fitted = fit_costmap([branch], route, [route], [0, 0], config)
    query = np.array([[5., 0., 0.], [20., 0., 2.]])
    symbolic = symbolic_cost(config)
    for point in query:
        actual = symbolic(point, fitted.centers.reshape(-1, 3).T, fitted.coefficients.ravel())
        np.testing.assert_allclose(float(actual), fitted.evaluate([point], config)[0], atol=1e-10)
    other_existence = Branch(branch.points, branch.times, .7, .9, 0, branch.dimensions)
    other = fit_costmap([other_existence], route, [route], [0, 0], config)
    assert not np.allclose(fitted.coefficients, other.coefficients)
    assert fitted.centers.shape == (6, 8, 3)


def test_footprint_collision_check():
    assert footprints_overlap([0, 0], 0, [4, 2], [3, 0], .2, [4, 2])
    assert not footprints_overlap([0, 0], 0, [4, 2], [0, 4], 0, [4, 2])


def test_polynomial_optimizer_and_3d_road():
    config = PlannerConfig(planning_mode='combined', horizon=0.5, gp_support_points=8, road_support_points=8,
                           solve_budget=.1, maximum_iterations=100, jit=False)
    route = np.array([[0., 0., 0.], [30., 0., 1.5]])
    road = Road(route, 3.85, 5.0)
    costmap = fit_costmap([], route, [route], [0, 0], config)
    optimizer = PolynomialMPC(config)
    plan = optimizer.solve([0, 0, 0, 0, 2, 0, 0], [0, 0], road, costmap)
    assert plan.coefficients.shape == (8,)
    assert plan.states.shape == (11, 7)
    assert plan.states[-1, 0] > 0
    np.testing.assert_allclose(plan.states[:, 2], plan.states[:, 0] * .05, atol=1e-8)
    assert np.isfinite(plan.commands).all()


def test_gp_branch_overflow_is_not_silent():
    config = PlannerConfig(horizon=.5, gp_support_points=8, road_support_points=8, jit=False)
    route = np.array([[0., 0., 0.], [30., 0., 0.]])
    branch = Branch(np.array([[10., 10., 0.], [11., 10., 0.]]), np.array([0., .5]), .5, .2, 0, (4., 2., 1.5))
    fitted = fit_costmap([branch] * 5, route, [route], [0, 0], config)
    assert fitted.omitted_branches == 1
    with pytest.raises(PlanningFailure, match='budget'):
        PolynomialMPC(config).solve([0, 0, 0, 0, 2, 0, 0], [0, 0], Road(route, 3.85, 5), fitted)


def test_deadline_miss_preserves_numerical_warm_start(monkeypatch):
    from plan_predictive_cost_map.planner import worker, costmap
    snapshot = SimpleNamespace(branches=(), road=SimpleNamespace(points=None), lanes=(),
                               state=np.zeros(7), previous_command=np.zeros(2), version=1, received_at=1.)
    optimizer = SimpleNamespace(warm_start=np.ones(8),
                                solve=lambda *args: SimpleNamespace(solve_seconds=.03))
    monkeypatch.setattr(costmap, 'fit_costmap', lambda *args: None)
    timestamps = iter([0., .002, .160])
    monkeypatch.setattr(worker, 'perf_counter', lambda: next(timestamps))
    update = worker.plan_snapshot(optimizer, snapshot, PlannerConfig())
    assert update['deadline_miss']
    assert 'plan' not in update
    assert update['total_ms'] == pytest.approx(160)
    assert update['validation_ms'] == pytest.approx(128)
    np.testing.assert_array_equal(optimizer.warm_start, np.ones(8))


def test_plan_over_control_period_but_within_deadline_is_accepted(monkeypatch):
    from plan_predictive_cost_map.planner import worker, costmap
    snapshot = SimpleNamespace(branches=(), road=SimpleNamespace(points=None), lanes=(),
                               state=np.zeros(7), previous_command=np.zeros(2), version=1, received_at=1.)
    plan = SimpleNamespace(solve_seconds=.05)
    optimizer = SimpleNamespace(solve=lambda *args: plan)
    monkeypatch.setattr(costmap, 'fit_costmap', lambda *args: None)
    timestamps = iter([0., .002, .080])
    monkeypatch.setattr(worker, 'perf_counter', lambda: next(timestamps))
    update = worker.plan_snapshot(optimizer, snapshot, PlannerConfig())
    assert update['plan'] is plan
    assert update['total_ms'] == pytest.approx(80)
    assert 'error' not in update


def test_delayed_warm_start_respects_actual_command_limits():
    config, vehicle = PlannerConfig(), VehicleConfig()
    coefficients = np.array([2.5, 6., -2., 1., .5, 2., -1., .5])
    previous = np.array([.25, .02])
    seed = command_seed(coefficients, previous, .12, config, vehicle)
    times = np.arange(config.steps) / config.steps
    basis = np.column_stack([np.ones_like(times), times, times**2, times**3])
    commands = np.column_stack([basis @ seed[:4], basis @ seed[4:]])
    rates = np.diff(np.vstack([previous, commands]), axis=0) / config.control_step
    assert np.max(np.abs(rates[:, 0])) <= config.maximum_jerk + 1e-8
    assert np.max(np.abs(rates[:, 1])) <= config.maximum_steering_rate + 1e-8
    assert np.max(commands[:, 0]) <= config.maximum_acceleration + 1e-8
    assert np.max(np.abs(commands[:, 1])) <= vehicle.maximum_steering + 1e-8


def test_drivable_union_allows_intentional_lane_change_but_not_departure():
    route = np.array([[0., 0., 0.], [50., 0., 0.]])
    target_area = box(-3., -1.925, 53., 1.925)
    adjacent_area = box(-3., 1.925, 53., 5.775)
    road = Road(route, 3.85, 10., target_area.union(adjacent_area))
    parameters = road.parameter_points(8)
    np.testing.assert_allclose(parameters[:, 3], 5.775)
    np.testing.assert_allclose(parameters[:, 4], 1.925)
    states = np.array([[10., 0., 0., 0., 5., 0., 0.], [12., 1.925, 0., 0., 5., 0., 0.],
                       [15., 3.85, 0., 0., 5., 0., 0.]])
    assert road.contains_trajectory(states, 4.5, 2., .2)
    states[-1, 1] = 6.
    assert not road.contains_trajectory(states, 4.5, 2., .2)


def test_longitudinal_mode_has_no_steering_decision_variables():
    config = PlannerConfig(planning_mode='longitudinal', horizon=.5, gp_support_points=8,
                           road_support_points=8, solve_budget=.1, jit=False)
    route = np.array([[0., 0., 0.], [50., 0., 0.]])
    costmap = fit_costmap([], route, [route], [0, 0], config)
    optimizer = PolynomialMPC(config)
    plan = optimizer.solve([0, 0, 0, 0, 50 / 3.6, 0, 0], [0, 0], Road(route, 3.85, 50 / 3.6), costmap)
    assert optimizer.decision_variables == 4
    assert plan.coefficients.shape == (4,)
    np.testing.assert_allclose(plan.commands[:, 1], 0, atol=1e-9)
    assert plan.states[-1, 0] > plan.states[0, 0]
    with pytest.raises(ValueError):
        PlannerConfig(planning_mode='unknown')


@pytest.mark.parametrize('speed', [0., 5., 50 / 3.6])
def test_internal_path_follower_matches_simulator_cpp(oracle, speed):
    points = np.array([[0., 0., 0., 1.925, 1.925], [5., 0., .5, 1.925, 1.925],
                       [10., 1., 1., 1.925, 1.925], [20., 5., 2., 1.925, 1.925],
                       [40., 5., 3., 1.925, 1.925]])
    expected = float(subprocess.check_output([str(oracle), '--follow', str(speed)], text=True))
    actual = float(internal_steering(5, VehicleConfig())([6., .3, .6, .1, speed, 0., 0.], .1, points.T))
    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize('pitch', [-.56, 0., .1])
def test_longitudinal_net_acceleration_compensates_road_loads(pitch):
    state = np.array([0., 0., 50., 0., 50 / 3.6, 0., 0.])
    requested = float(acceleration_command()(state, 0., 0., pitch))
    _, accelerations = dynamic_step(.05)(state, [requested, 0.], [pitch, pitch])
    assert abs(float(accelerations[0])) < .02
    assert -9.81 <= requested <= 9.81


def test_new_virtual_object_only_invalidates_an_intersecting_plan():
    states = np.array([[0., 0., 0., 0., 5., 0., 0.], [5., 0., 0., 0., 5., 0., 0.]])
    times = np.array([0., 1.])
    far = Branch(np.array([[100., 0., 0.], [100., 0., 0.]]), times, .9, .02, 1, (.6, .6, 1.8), True, 0., b'far')
    near = Branch(np.array([[5., 0., 0.], [5., 0., 0.]]), times, .9, .5, 1, (.6, .6, 1.8), True, 0., b'near')
    assert not new_virtual_hazard([far], set(), states, times, [4.5, 2.], .2)
    assert new_virtual_hazard([near], set(), states, times, [4.5, 2.], .2)
    assert not new_virtual_hazard([near], {b'near'}, states, times, [4.5, 2.], .2)


def test_longitudinal_execution_uses_measured_progress_not_wall_time():
    states = np.zeros((8, 7))
    states[:, 0] = np.arange(8) * .6
    states[:, 4] = 50 / 3.6
    measured = states[1].copy()
    assert longitudinal_tracking_index(states, measured) == 1
    assert longitudinal_tracking_index(states, measured, minimum_index=2) == 2


def test_objective_has_distinct_matlab_style_terms():
    config, vehicle = PlannerConfig(), VehicleConfig()
    fast = objective_terms(12., 10., [1., 2.], [.5, .1], [.2, .05], -.5, config, vehicle)
    slow = objective_terms(8., 10., [-2., 0.], [0., 0.], [0., 0.], -.5, config, vehicle)
    assert float(fast['overspeed']) > 0 and float(fast['underspeed']) == 0
    assert float(slow['underspeed']) > 0 and float(slow['overspeed']) == 0
    assert float(fast['positive_acceleration']) > 0 and float(fast['braking_acceleration']) == 0
    assert float(slow['braking_acceleration']) > 0 and float(slow['positive_acceleration']) == 0
    assert float(fast['lateral_acceleration']) > 0
    assert float(fast['steering_amplitude']) > 0 and float(fast['input_rates']) > 0


def test_longitudinal_reuses_a_feasible_seed_after_solver_interrupt():
    config = PlannerConfig(planning_mode='longitudinal', horizon=.5, gp_support_points=8, road_support_points=8, jit=False)
    route = np.array([[0., 0., 0.], [50., 0., 0.]])
    costmap = fit_costmap([], route, [route], [0, 0], config)
    optimizer = PolynomialMPC(config)

    class InterruptedSolver:
        def __call__(self, **arguments):
            return {'x': np.array([100., 0., 0., 0.])}

        def stats(self):
            return {'success': False, 'return_status': 'Maximum_WallTime_Exceeded', 'iter_count': 1}

    optimizer.solver = InterruptedSolver()
    plan = optimizer.solve([0, 0, 0, 0, 10, 0, 0], [0, 0], Road(route, 3.85, 13.9), costmap)
    assert plan.status.endswith(':Validated_Warm_Start')
    assert np.isfinite(plan.states).all()