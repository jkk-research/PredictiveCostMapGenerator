"""Acceptance checks must reject unsafe, stationary and off-road behavior."""

import os
from pathlib import Path

import numpy as np
import pytest

from scenario_acceptance import AcceptanceMonitor, DrivingSample, ScenarioTruth, TruthActor, run_case, warm_solver


def straight_scene(actors=()):
    route = np.array([[0., 0., 0.], [30., 0., 0.], [60., 0., 0.]])
    lanes = [route, route + [0., 3.85, 0.]]
    return ScenarioTruth(Path('synthetic.mat'), np.arange(0, 10.05, .05), route, lanes,
                         np.full(3, 10.), list(actors))


def sample(elapsed=0., position=(5., 0., 0.), speed=5., acceleration=0., lateral=0.):
    return DrivingSample(elapsed, np.array(position), 0., speed, acceleration, lateral, 0.)


def test_no_progress_is_not_a_pass():
    monitor = AcceptanceMonitor(straight_scene())
    monitor.observe(sample(speed=0.))
    report = monitor.report(30., 'timeout')
    assert report['status'] == 'FAIL'
    assert any(item['criterion'] == 'progress' for item in report['failures'])


def test_early_failure_does_not_claim_unobserved_behavior_was_tested():
    monitor = AcceptanceMonitor(straight_scene())
    monitor.observe(sample(position=(5., 10., 0.)))
    report = monitor.report(.05, 'first_acceptance_failure')
    assert report['status'] == 'FAIL'
    assert report['coverage']['progress_status'] == 'NOT_EVALUATED'
    assert not any(item['criterion'] == 'progress' for item in report['failures'])


def test_overtaking_is_not_mislabeled_as_lane_departure():
    monitor = AcceptanceMonitor(straight_scene())
    monitor.observe(sample(position=(10., 3.85, 0.)))
    assert not monitor.failures
    assert monitor.excursion
    monitor.observe(sample(.05, position=(12., 7., 0.)))
    assert any(item['criterion'] == 'drivable_surface' for item in monitor.failures)


def test_hidden_actor_collision_is_detected_without_scenario_messages():
    actor = TruthActor(np.array([[5., 0., 0.]]), np.array([0.]), np.array([1.]),
                       np.array([4., 2., 1.5]), False)
    monitor = AcceptanceMonitor(straight_scene([actor]))
    monitor.observe(sample())
    assert any(item['criterion'] == 'collision' for item in monitor.failures)


def test_comfort_and_pedestrian_yield_are_checked():
    actor = TruthActor(np.tile([8., 0., 0.], (201, 1)), np.zeros(201), np.ones(201),
                       np.array([.6, .6, 1.8]), True)
    monitor = AcceptanceMonitor(straight_scene([actor]))
    monitor.observe(sample(position=(3., 0., 0.), speed=0.))
    assert monitor.pedestrian_stops == 1
    monitor.observe(sample(.05, position=(5., 0., 0.), acceleration=3., lateral=4.))
    assert any(item['criterion'] == 'smoothness' for item in monitor.failures)
    assert any(item['criterion'] == 'curve_speed' for item in monitor.failures)
    monitor.observe(sample(.1, position=(5.5, 0., 0.)))
    assert any(item['criterion'] == 'pedestrian_yield' for item in monitor.failures)


def test_unnecessary_stops_and_lane_drift_fail():
    monitor = AcceptanceMonitor(straight_scene())
    monitor.observe(sample(speed=2.))
    monitor.observe(sample(1., speed=0.))
    monitor.observe(sample(2., position=(6., 0., 0.), speed=2.))
    assert any(item['criterion'] == 'stop_restart' for item in monitor.failures)
    monitor = AcceptanceMonitor(straight_scene())
    monitor.observe(sample(position=(10., 3.85, 0.)))
    monitor.observe(sample(4., position=(20., 3.85, 0.)))
    assert any(item['criterion'] == 'lane_discipline' for item in monitor.failures)


def test_virtual_hazard_can_justify_waiting_and_an_excursion():
    monitor = AcceptanceMonitor(straight_scene())
    first = sample(speed=2., position=(10., 3.85, 0.))
    first.uncertain_hazards = True
    monitor.observe(first)
    stopped = sample(1., speed=0., position=(10., 3.85, 0.))
    stopped.uncertain_hazards = True
    monitor.observe(stopped)
    resumed = sample(5., speed=2., position=(15., 3.85, 0.))
    resumed.uncertain_hazards = True
    monitor.observe(resumed)
    assert not any(item['criterion'] in ('stop_restart', 'lane_discipline') for item in monitor.failures)


ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = sorted((ROOT / 'src/scenario_files').glob('*.mat'))


@pytest.mark.parametrize('scenario', SCENARIOS, ids=lambda path: path.stem)
def test_every_mat_fixture_has_a_ground_truth_oracle(scenario):
    truth = ScenarioTruth.load(scenario)
    assert truth.duration > 0
    assert truth.line.length > 0
    assert truth.road_area.is_valid
    assert len(truth.actors) > 0


@pytest.fixture(scope='session')
def prepared_acceptance_solver():
    warm_solver()


@pytest.mark.skipif(os.environ.get('PCM_SCENARIO_ACCEPTANCE') != '1', reason='Full driving acceptance is opt-in')
@pytest.mark.parametrize('scenario', SCENARIOS, ids=lambda path: path.stem)
def test_full_driving_scenario(scenario, prepared_acceptance_solver):
    report = run_case(scenario, ROOT)
    assert report['status'] == 'PASS', report