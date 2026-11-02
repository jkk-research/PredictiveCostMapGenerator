"""Short-horizon linearized MPC for trajectory tracking; no GP optimization."""

from dataclasses import dataclass
from time import perf_counter

import casadi as ca
import numpy as np
import osqp
from scipy import sparse

from .config import TrackerConfig, VehicleConfig
from .dynamics import acceleration_command, dynamic_step


class TrackingFailure(RuntimeError):
    """No numerically valid tracking command is available."""


@dataclass(frozen=True)
class TrackingReference:
    stamp_ns: int
    times: np.ndarray
    states: np.ndarray
    pitch: np.ndarray
    acceleration: np.ndarray
    steering: np.ndarray

    def __post_init__(self):
        count = len(self.times)
        if count < 2 or self.states.shape != (count, 7):
            raise ValueError('Tracking reference needs at least two seven-state samples')
        if np.any(np.diff(self.times) <= 0) or self.times[0] < 0:
            raise ValueError('Reference times must be strictly increasing and nonnegative')
        for values in (self.times, self.states, self.pitch, self.acceleration, self.steering):
            if len(values) != count or not np.isfinite(values).all():
                raise ValueError('Nonfinite tracking reference')
        if np.any(self.states[:, 4] < 0):
            raise ValueError('Reverse trajectories are not supported')

    def sample(self, times, measured_yaw=None):
        samples = np.asarray(times).reshape(-1)
        states = np.column_stack([np.interp(samples, self.times, self.states[:, index]) for index in range(7)])
        if measured_yaw is not None:
            states[:, 3] += 2 * np.pi * round((measured_yaw - states[0, 3]) / (2 * np.pi))
        return (states, np.interp(samples, self.times, self.pitch),
                np.interp(samples, self.times, self.acceleration),
                np.interp(samples, self.times, self.steering))


@dataclass(frozen=True)
class TrackingResult:
    command: np.ndarray
    states: np.ndarray
    solve_seconds: float
    iterations: int
    status: str
    net_acceleration: float


class TrackingMPC:
    def __init__(self, config=TrackerConfig(), vehicle=VehicleConfig()):
        self.config, self.vehicle = config, vehicle
        self.step = dynamic_step(config.time_step, vehicle)
        self.feedforward = acceleration_command(vehicle)
        state = ca.SX.sym('state', 7)
        command = ca.SX.sym('command', 2)
        road = ca.SX.sym('road', 2)
        next_state, _ = self.step(state, command, road)
        self.linearize = ca.Function('tracking_linearization', [state, command, road],
                                     [ca.jacobian(next_state, state), ca.jacobian(next_state, command), next_state])
        count = config.steps
        self.state_size = 7 * (count + 1)
        self.variable_count = self.state_size + 2 * count
        state_weights = np.array([1., 20., 1., 12., 6., 1., 1.])
        diagonal = np.r_[np.tile(state_weights, count), 5 * state_weights, np.tile([.1, 1.], count)]
        self.rate_matrix = sparse.diags([np.ones(2 * count), -np.ones(2 * count - 2)],
                                         [0, -2], shape=(2 * count, 2 * count), format='csc')
        self.rate_weights = sparse.diags(np.tile([.2, 4.], count), format='csc')
        state_hessian = sparse.diags(diagonal[:self.state_size], format='csc')
        input_hessian = (sparse.diags(diagonal[self.state_size:], format='csc') +
                         self.rate_matrix.T @ self.rate_weights @ self.rate_matrix)
        self.hessian = sparse.triu(2 * sparse.block_diag([state_hessian, input_hessian], format='csc'), format='csc')
        self.solver = None
        self.pattern = None

    def solve(self, measured, reference, elapsed, previous_net_acceleration=0., previous_steering=0.):
        config, vehicle = self.config, self.vehicle
        started = perf_counter()
        measured = np.asarray(measured, dtype=float)
        if measured.shape != (7,) or not np.isfinite(measured).all():
            raise TrackingFailure('Invalid measured vehicle state')
        if elapsed < -config.time_step or elapsed + config.horizon > reference.times[-1] + 1e-6:
            raise TrackingFailure('Reference has insufficient remaining horizon')
        times = elapsed + np.arange(config.steps + 1) * config.time_step
        states, pitch, net_acceleration, steering = reference.sample(times, measured[3])
        feedforward = np.array([[float(self.feedforward(states[index], net_acceleration[index],
                                                       steering[index], pitch[index])), steering[index]]
                                for index in range(config.steps)])
        initial_error = measured - states[0]
        initial_error[3] = np.arctan2(np.sin(initial_error[3]), np.cos(initial_error[3]))
        rows, columns, data = [], [], []
        lower, upper = [], []

        def entry(row, column, value):
            rows.append(row)
            columns.append(column)
            data.append(float(value))

        for axis in range(7):
            entry(axis, axis, 1)
        lower.extend(initial_error)
        upper.extend(initial_error)
        for index in range(config.steps):
            matrix_a, matrix_b, nominal = self.linearize(states[index], feedforward[index], [pitch[index], pitch[index]])
            matrix_a, matrix_b = np.array(matrix_a), np.array(matrix_b)
            defect = np.asarray(nominal).ravel() - states[index + 1]
            defect[3] = np.arctan2(np.sin(defect[3]), np.cos(defect[3]))
            base = 7 * (index + 1)
            for axis in range(7):
                entry(base + axis, base + axis, 1)
                for state_axis in range(7):
                    entry(base + axis, 7 * index + state_axis, -matrix_a[axis, state_axis])
                for input_axis in range(2):
                    entry(base + axis, self.state_size + 2 * index + input_axis, -matrix_b[axis, input_axis])
            lower.extend(defect)
            upper.extend(defect)
        row = len(lower)
        for index in range(config.steps):
            for axis, minimum, maximum in ((0, -vehicle.maximum_deceleration, vehicle.maximum_acceleration),
                                            (1, -vehicle.maximum_steering, vehicle.maximum_steering)):
                entry(row, self.state_size + 2 * index + axis, 1)
                lower.append(minimum - feedforward[index, axis])
                upper.append(maximum - feedforward[index, axis])
                row += 1
        for index in range(config.steps):
            entry(row, self.state_size + 2 * index, 1)
            lower.append(-config.maximum_deceleration - net_acceleration[index])
            upper.append(config.maximum_acceleration - net_acceleration[index])
            row += 1
        nominal_control = np.column_stack([net_acceleration[:-1], steering[:-1]])
        nominal_changes = np.diff(np.vstack([[previous_net_acceleration, previous_steering], nominal_control]), axis=0)
        for index in range(config.steps):
            for axis, limit in ((0, config.maximum_jerk), (1, config.maximum_steering_rate)):
                entry(row, self.state_size + 2 * index + axis, 1)
                if index:
                    entry(row, self.state_size + 2 * (index - 1) + axis, -1)
                lower.append(-limit * config.time_step - nominal_changes[index, axis])
                upper.append(limit * config.time_step - nominal_changes[index, axis])
                row += 1
        for index in range(1, config.steps + 1):
            entry(row, 7 * index + 4, 1)
            lower.append(-states[index, 4])
            upper.append(config.maximum_speed - states[index, 4])
            row += 1
        constraints = sparse.coo_matrix((data, (rows, columns)), shape=(row, self.variable_count)).tocsc()
        linear = np.zeros(self.variable_count)
        linear[self.state_size:] = np.asarray(2 * self.rate_matrix.T @ self.rate_weights @ nominal_changes.ravel()).ravel()
        lower, upper = np.asarray(lower), np.asarray(upper)
        pattern = (tuple(constraints.indptr), tuple(constraints.indices))
        if self.solver is None or self.pattern != pattern:
            self.solver = osqp.OSQP()
            self.solver.setup(P=self.hessian, q=linear, A=constraints, l=lower, u=upper,
                              verbose=False, warm_starting=True, polishing=False,
                              eps_abs=1e-4, eps_rel=1e-4, max_iter=config.maximum_iterations,
                              time_limit=config.solve_budget)
            self.pattern = pattern
        else:
            self.solver.update(q=linear, Ax=constraints.data, l=lower, u=upper)
        result = self.solver.solve(raise_error=False)
        if result.x is None or result.info.status_val not in (1, 2) or not np.isfinite(result.x).all():
            raise TrackingFailure('Tracking QP: ' + result.info.status)
        values = constraints @ result.x
        violation = max(float(np.max(lower - values)), float(np.max(values - upper)), 0.)
        if violation > .002:
            raise TrackingFailure('Tracking QP constraint violation: %.4f' % violation)
        command = feedforward[0] + result.x[self.state_size:self.state_size + 2]
        predicted = states + result.x[:self.state_size].reshape(-1, 7)
        _, acceleration = self.step(measured, command, [pitch[0], pitch[0]])
        return TrackingResult(command, predicted, perf_counter() - started, result.info.iter,
                              result.info.status, float(acceleration[0]))