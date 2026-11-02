"""Tracking tests use generated references, independent of the GP planner."""

import numpy as np
import pytest

from plan_predictive_cost_map.planner.config import TrackerConfig
from plan_predictive_cost_map.planner.dynamics import dynamic_step
from plan_predictive_cost_map.planner.tracking import TrackingFailure, TrackingMPC, TrackingReference


def constant_speed_reference(speed=50 / 3.6, pitch=0., duration=5.):
    times = np.arange(0, duration + .025, .05)
    states = np.zeros((len(times), 7))
    states[:, 0] = speed * times * np.cos(pitch)
    states[:, 2] = 50 + speed * times * np.sin(pitch)
    states[:, 4] = speed
    return TrackingReference(0, times, states, np.full(len(times), pitch),
                              np.zeros(len(times)), np.zeros(len(times)))


@pytest.mark.parametrize('pitch', [0., -.56, .1])
def test_tracks_50_kph_without_unnecessary_downhill_braking(pitch):
    reference = constant_speed_reference(pitch=pitch)
    tracker = TrackingMPC()
    step = dynamic_step(.05)
    measured = reference.states[0].copy()
    net_acceleration = 0.
    previous_steering = 0.
    errors, runtimes = [], []
    for index in range(50):
        result = tracker.solve(measured, reference, index * .05, net_acceleration, previous_steering)
        measured_next, acceleration = step(measured, result.command, [pitch, pitch])
        measured = np.asarray(measured_next).ravel()
        net_acceleration, previous_steering = float(acceleration[0]), result.command[1]
        errors.append(abs(measured[4] - 50 / 3.6))
        runtimes.append(result.solve_seconds)
        assert np.isfinite(measured).all()
        assert -9.81 <= result.command[0] <= 9.81
    assert max(errors) < .1
    assert abs(net_acceleration) < .1
    print('Tracker pitch', pitch, 'peak speed error', max(errors), 'p95 ms', np.percentile(runtimes, 95) * 1000)


def test_tracker_does_not_need_fresh_replans_every_control_tick():
    tracker = TrackingMPC()
    reference = constant_speed_reference()
    result = tracker.solve(reference.states[20], reference, 1.)
    assert abs(result.net_acceleration) < .1
    with pytest.raises(TrackingFailure, match='remaining horizon'):
        tracker.solve(reference.states[-1], reference, 4.9)


def test_tracker_reference_validation():
    reference = constant_speed_reference()
    with pytest.raises(ValueError):
        TrackingReference(0, np.zeros_like(reference.times), reference.states,
                          reference.pitch, reference.acceleration, reference.steering)


def test_tracker_corrects_lateral_and_speed_offsets():
    reference = constant_speed_reference()
    tracker = TrackingMPC()
    step = dynamic_step(.05)
    measured = reference.states[0].copy()
    measured[1] = .3
    measured[4] -= .5
    previous_acceleration, previous_steering = 0., 0.
    for index in range(40):
        result = tracker.solve(measured, reference, index * .05, previous_acceleration, previous_steering)
        next_state, acceleration = step(measured, result.command, [0., 0.])
        measured = np.asarray(next_state).ravel()
        previous_acceleration, previous_steering = float(acceleration[0]), result.command[1]
    assert abs(measured[1]) < .15
    assert abs(measured[4] - 50 / 3.6) < .3