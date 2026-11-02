"""Fixed-graph polynomial-command MPC with external GP posterior fitting."""

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import platform
from time import perf_counter

import casadi as ca
import numpy as np

from .config import PlannerConfig, VehicleConfig
from .costmap import symbolic_cost
from .dynamics import acceleration_command, dynamic_step
from .geometry import resample_polyline, trajectory_overlaps


class PlanningFailure(RuntimeError):
    """No validated solution is available for this planning cycle."""


def command_seed(coefficients, previous, interval, config, vehicle):
    seed = np.zeros(8)
    fraction = interval / config.horizon
    for offset in (0, 4):
        for power in range(4):
            seed[offset + power] = sum(coefficients[offset + degree] * math.comb(degree, power) *
                                      fraction**(degree - power) for degree in range(power, 4))
    times = np.arange(config.steps) / config.steps
    basis = np.column_stack([times, times**2, times**3])
    derivative = np.column_stack([np.ones_like(times), 2 * times, 3 * times**2]) / config.horizon
    bounds = ((-config.maximum_deceleration, config.maximum_acceleration, config.maximum_jerk),
              (-vehicle.maximum_steering, vehicle.maximum_steering, config.maximum_steering_rate))
    for control, (minimum, maximum, rate) in enumerate(bounds):
        offset = 4 * control
        seed[offset] = np.clip(seed[offset], previous[control] - rate * config.control_step,
                               previous[control] + rate * config.control_step)
        seed[offset] = np.clip(seed[offset], minimum, maximum)
        deviation = basis @ seed[offset + 1:offset + 4]
        variation = np.max(np.abs(derivative @ seed[offset + 1:offset + 4]))
        scale = min(1., rate / max(variation, 1e-12))
        if np.max(deviation) > 0:
            scale = min(scale, (maximum - seed[offset]) / np.max(deviation))
        if np.min(deviation) < 0:
            scale = min(scale, (minimum - seed[offset]) / np.min(deviation))
        seed[offset + 1:offset + 4] *= max(0., scale)
    return seed


@dataclass(frozen=True)
class Plan:
    states: np.ndarray
    commands: np.ndarray
    accelerations: np.ndarray
    coefficients: np.ndarray
    solve_seconds: float
    iterations: int
    status: str
    objective_components: dict = None


def objective_terms(speed, target_speed, accelerations, command, rates, risk, config, vehicle):
    overspeed = ca.fmax(0, speed - target_speed)
    underspeed = ca.fmax(0, target_speed - speed)
    tolerance = config.overspeed_tolerance
    return {
        'gp_risk': config.weight_safety * (risk + 1)**2,
        'overspeed': config.weight_velocity * config.weight_overspeed *
            (1 + tolerance**4) * overspeed**4 / (overspeed**4 + tolerance**4),
        'underspeed': config.weight_velocity * config.weight_underspeed *
            (underspeed / ca.fmax(target_speed, 1))**2,
        'positive_acceleration': config.weight_acceleration * config.weight_positive_acceleration *
            (ca.fmax(accelerations[0], 0) / config.maximum_acceleration)**2,
        'braking_acceleration': config.weight_acceleration * config.weight_braking_acceleration *
            (ca.fmin(accelerations[0], 0) / config.maximum_deceleration)**2,
        'lateral_acceleration': config.weight_lateral_acceleration *
            (accelerations[1] / config.maximum_lateral_acceleration)**2,
        'steering_amplitude': config.weight_steering * (command[1] / vehicle.maximum_steering)**2,
        'drive_amplitude': config.weight_input_amplitude * (command[0] / vehicle.maximum_acceleration)**2,
        'input_rates': config.weight_rate * ((rates[0] / config.maximum_jerk)**2 +
            (rates[1] / config.maximum_steering_rate)**2),
        'crash': config.weight_crash * ca.fmax(0, risk - .2)**2,
    }


def road_lookup(count):
    query = ca.SX.sym('query', 3)
    points = ca.SX.sym('road_points', 5, count)
    minimum = 1e30
    result = ca.SX.zeros(9)
    remaining_lengths = [ca.SX(0)] * count
    for index in reversed(range(count - 1)):
        remaining_lengths[index] = remaining_lengths[index + 1] + ca.norm_2(points[:2, index + 1] - points[:2, index])
    for index in range(count - 1):
        first, delta = points[:3, index], points[:3, index + 1] - points[:3, index]
        squared_length = ca.fmax(ca.dot(delta[:2], delta[:2]), 1e-12)
        fraction = ca.fmin(1, ca.fmax(0, ca.dot(query[:2] - first[:2], delta[:2]) / squared_length))
        projection = first + fraction * delta
        distance = ca.sumsqr(query[:2] - projection[:2])
        heading = ca.atan2(delta[1], delta[0])
        pitch = ca.atan(delta[2] / ca.sqrt(squared_length) * ca.cos(query[2] - heading))
        lateral = -(query[0] - projection[0]) * ca.sin(heading) + (query[1] - projection[1]) * ca.cos(heading)
        remaining = remaining_lengths[index + 1] + (1 - fraction) * ca.sqrt(squared_length)
        widths = points[3:, index] + fraction * (points[3:, index + 1] - points[3:, index])
        result = ca.if_else(distance < minimum, ca.vertcat(projection, heading, pitch, lateral, remaining, widths), result)
        minimum = ca.fmin(minimum, distance)
    return ca.Function('road_lookup', [query, points], [result])


def internal_steering(count, settings):
    state = ca.SX.sym('state', 7)
    pitch = ca.SX.sym('pitch')
    points = ca.SX.sym('points', 5, count)
    nearest_distance = 1e30
    nearest_arc = ca.SX(0)
    arc = [ca.SX(0)]
    for index in range(count):
        if index:
            arc.append(arc[-1] + ca.norm_2(points[:3, index] - points[:3, index - 1]))
        distance = ca.sumsqr(state[:2] - points[:2, index])
        nearest_arc = ca.if_else(distance < nearest_distance, arc[index], nearest_arc)
        nearest_distance = ca.fmin(nearest_distance, distance)
    desired_arc = nearest_arc + ca.fmax(settings.minimum_lookahead, settings.lookahead_time * state[4])
    target = points[:3, -1]
    for index in reversed(range(count)):
        target = ca.if_else(arc[index] >= desired_arc, points[:3, index], target)
    delta = target - state[:3]
    local_y = -delta[0] * ca.sin(state[3]) + delta[1] * ca.cos(state[3])
    local_x = ca.cos(pitch) * (delta[0] * ca.cos(state[3]) + delta[1] * ca.sin(state[3])) + ca.sin(pitch) * delta[2]
    squared_distance = local_x**2 + local_y**2
    angle = ca.atan2(2 * settings.wheelbase * local_y, ca.fmax(squared_distance, 1e-12))
    angle = ca.if_else(squared_distance < 1e-8, 0, angle)
    return ca.Function('internal_steering', [state, pitch, points],
                       [ca.fmin(settings.maximum_steering, ca.fmax(-settings.maximum_steering, angle))])


class PolynomialMPC:
    def __init__(self, config=PlannerConfig(), vehicle=VehicleConfig()):
        self.config = config
        self.vehicle = vehicle
        self.warm_start = None
        self.step = dynamic_step(config.control_step, vehicle)
        self.lookup = road_lookup(config.road_support_points)
        self.longitudinal = config.planning_mode == 'longitudinal'
        self.decision_variables = 4 if self.longitudinal else 8
        self.path_follower = internal_steering(config.road_support_points, vehicle)
        self.acceleration_command = acceleration_command(vehicle)
        coefficients = ca.MX.sym('polynomial', self.decision_variables)
        parameters = ca.MX.sym('parameters', 7 + 2 + 2 + 5 * config.road_support_points +
                               24 * config.gp_support_points)
        state = parameters[:7]
        previous = parameters[7:9]
        width, target_speed = parameters[9], parameters[10]
        road_end = 11 + 5 * config.road_support_points
        road = ca.reshape(parameters[11:road_end], 5, config.road_support_points)
        centers_end = road_end + 18 * config.gp_support_points
        centers = ca.reshape(parameters[road_end:centers_end], 3, 6 * config.gp_support_points)
        gp_coefficients = parameters[centers_end:]
        cost_function = symbolic_cost(config)
        states, commands, accelerations = [state], [], []
        constraints, lower, upper = [], [], []
        objective = 0
        components = {}

        def constrain(value, minimum, maximum):
            constraints.append(value)
            lower.extend(np.broadcast_to(minimum, (value.numel(),)))
            upper.extend(np.broadcast_to(maximum, (value.numel(),)))

        for index in range(config.steps):
            normalized_time = index / config.steps
            basis = ca.vertcat(1, normalized_time, normalized_time**2, normalized_time**3)
            surface = self.lookup(ca.vertcat(state[0], state[1], state[3]), road)
            steering = self.path_follower(state, surface[4], road) if self.longitudinal else ca.dot(coefficients[4:], basis)
            desired_acceleration = ca.dot(coefficients[:4], basis)
            requested_acceleration = self.acceleration_command(state, desired_acceleration, steering, surface[4])
            command = ca.vertcat(requested_acceleration, steering)
            constrain(desired_acceleration, -config.maximum_deceleration, config.maximum_acceleration)
            constrain(steering, -vehicle.maximum_steering, vehicle.maximum_steering)
            rates = (ca.vertcat(desired_acceleration, steering) - previous) / config.control_step
            if self.longitudinal:
                constrain(rates[0], -config.maximum_jerk, config.maximum_jerk)
            else:
                constrain(rates, [-config.maximum_jerk, -config.maximum_steering_rate],
                          [config.maximum_jerk, config.maximum_steering_rate])
            state, acceleration = self.step(state, command, ca.vertcat(surface[4], surface[4]))
            surface = self.lookup(ca.vertcat(state[0], state[1], state[3]), road)
            state = ca.vertcat(state[0], state[1], surface[2], state[3:])
            heading_error = state[3] - surface[3]
            radius = (0.5 * config.ego_length * ca.fabs(ca.sin(heading_error)) +
                      0.5 * config.ego_width * ca.fabs(ca.cos(heading_error)) + config.safety_margin)
            constrain(surface[5] + radius - surface[7], -np.inf, 0)
            constrain(-surface[5] + radius - surface[8], -np.inf, 0)
            constrain(state[4], 0, config.maximum_speed)
            stopping_distance = state[4]**2 / (2 * config.maximum_deceleration)
            constrain(stopping_distance + config.ego_length / 2 + config.safety_margin - surface[6], -np.inf, 0)
            constrain(acceleration[1], -config.maximum_lateral_acceleration, config.maximum_lateral_acceleration)
            constrain(acceleration[0], -config.maximum_deceleration, config.maximum_acceleration)
            stride = round(config.cost_step / config.control_step)
            if (index + 1) % stride == 0 or index == config.steps - 1:
                risk = cost_function(ca.vertcat(state[0], state[1], (index + 1) * config.control_step),
                                     centers, gp_coefficients)
                constrain(risk, -np.inf, config.risk_limit)
                terms = objective_terms(state[4], target_speed, acceleration, command, rates, risk, config, vehicle)
                for name in ('gp_risk', 'overspeed', 'underspeed', 'crash'):
                    components[name] = components.get(name, 0) + stride * terms[name] / config.steps
            terms = objective_terms(state[4], target_speed, acceleration, command, rates, 0, config, vehicle)
            for name in ('positive_acceleration', 'braking_acceleration', 'lateral_acceleration',
                         'steering_amplitude', 'drive_amplitude', 'input_rates'):
                components[name] = components.get(name, 0) + terms[name] / config.steps
            states.append(state)
            commands.append(command)
            accelerations.append(acceleration)
            previous = ca.vertcat(desired_acceleration, steering)
        components['terminal'] = config.weight_terminal * (risk + 1)**2
        objective = sum(components.values())
        self.component_names = tuple(components)
        self.objective_evaluate = ca.Function('objective_components', [coefficients, parameters],
                              [ca.vertcat(*components.values())])
        stacked_constraints = ca.vertcat(*constraints)
        self.evaluate = ca.Function('evaluate_plan', [coefficients, parameters],
                                    [ca.horzcat(*states), ca.horzcat(*commands),
                                     ca.horzcat(*accelerations), stacked_constraints])
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        options = {
            'print_time': False, 'error_on_fail': False,
            'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'ipopt.tol': 1e-3,
            'ipopt.constr_viol_tol': 1e-4, 'ipopt.acceptable_tol': 5e-3,
            'ipopt.max_iter': config.maximum_iterations, 'ipopt.max_wall_time': config.solve_budget,
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.warm_start_init_point': 'yes',
        }
        if config.jit:
            options.update(jit=True, compiler='shell',
                           jit_name='pcm_planner',
                           jit_serialize='embed',
                           jit_options={'flags': ['-O1']})
        signature = repr((config, vehicle, ca.__version__, platform.machine())).encode()
        for name in ('optimizer.py', 'dynamics.py', 'costmap.py', 'config.py'):
            signature += Path(__file__).with_name(name).read_bytes()
        cache = Path.home() / '.cache' / 'pcm_planner' / (hashlib.sha256(signature).hexdigest() + '.casadi')
        self.solver = None
        if config.jit and cache.is_file():
            try:
                self.solver = ca.Function.load(str(cache))
            except RuntimeError:
                pass
        if self.solver is None:
            self.solver = ca.nlpsol('pcm_mpc', 'ipopt',
                                    {'x': coefficients, 'p': parameters, 'f': objective, 'g': stacked_constraints}, options)
            if config.jit:
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache.with_suffix('.' + str(os.getpid()) + '.tmp')
                    self.solver.save(str(temporary))
                    temporary.replace(cache)
                except OSError:
                    pass

    def solve(self, initial, previous_command, road, costmap, branches=(), replan_interval=None):
        if costmap.omitted_branches:
            raise PlanningFailure('Prediction branch budget exceeded; refusing an incomplete cost map')
        road_points = road.parameter_points(self.config.road_support_points)
        parameters = np.r_[initial, previous_command, road.width, min(road.speed, self.config.maximum_speed),
                           road_points.ravel(), costmap.centers.ravel(), costmap.coefficients.ravel()]
        if not np.isfinite(parameters).all() or road.width <= self.config.ego_width or road.speed < 0:
            raise PlanningFailure('Invalid model, road or GP input')
        seed = np.zeros(8) if self.warm_start is None else self.warm_start.copy()
        if self.warm_start is None:
            seed[0] = min(previous_command[0] + self.config.maximum_jerk * self.config.control_step, 0.25)
            seed[1] = min(6.0, self.config.maximum_jerk * self.config.horizon,
                          self.config.maximum_acceleration - seed[0]) if initial[4] < .05 else 0.0
            seed[4] = previous_command[1]
        seed = command_seed(seed, previous_command,
                        0. if self.warm_start is None else (self.config.control_step if replan_interval is None else replan_interval),
                        self.config, self.vehicle)
        seed = seed[:self.decision_variables]
        started = perf_counter()
        result = self.solver(x0=seed, p=parameters, lbg=self.lower, ubg=self.upper,
                                lbx=[-self.config.maximum_deceleration, -100, -100, -100,
                                    -self.vehicle.maximum_steering, -10, -10, -10][:self.decision_variables],
                                ubx=[self.config.maximum_acceleration, 100, 100, 100,
                                    self.vehicle.maximum_steering, 10, 10, 10][:self.decision_variables])
        elapsed = perf_counter() - started
        coefficients = np.asarray(result['x']).ravel()
        stats = self.solver.stats()
        states, commands, accelerations, constraints = [np.asarray(value) for value in self.evaluate(coefficients, parameters)]
        constraints = constraints.ravel()
        violation = max(np.max(self.lower - constraints), np.max(constraints - self.upper), 0)
        status = stats.get('return_status', 'unknown')
        usable_status = stats.get('success') or status in ('Maximum_WallTime_Exceeded', 'Maximum_Iterations_Exceeded')
        if self.longitudinal and usable_status and (not np.isfinite(constraints).all() or violation > 1e-3):
            seed_states, seed_commands, seed_accelerations, seed_constraints = [np.asarray(value) for value in self.evaluate(seed, parameters)]
            seed_constraints = seed_constraints.ravel()
            seed_violation = max(np.max(self.lower - seed_constraints), np.max(seed_constraints - self.upper), 0)
            if np.isfinite(seed_states).all() and np.isfinite(seed_commands).all() and np.isfinite(seed_constraints).all() and seed_violation <= 1e-3:
                states, commands, accelerations, constraints = seed_states, seed_commands, seed_accelerations, seed_constraints
                coefficients, violation = seed.copy(), seed_violation
                status += ':Validated_Warm_Start'
        if not usable_status or not np.isfinite(states).all() or not np.isfinite(commands).all() or not np.isfinite(constraints).all() or violation > 1e-3:
            raise PlanningFailure(str(stats.get('return_status')) + '; constraint violation=' + str(violation))
        if not road.contains_trajectory(states.T, self.config.ego_length, self.config.ego_width, self.config.safety_margin):
            raise PlanningFailure('Trajectory leaves the permitted drivable surface')
        times = np.arange(self.config.steps + 1) * self.config.control_step
        risks = costmap.evaluate(np.column_stack([states[0], states[1], times]), self.config)
        if not np.isfinite(risks).all() or np.max(risks[1:]) > self.config.risk_limit + 1e-3:
            raise PlanningFailure('Dense GP validation rejected the trajectory')
        for branch in branches:
            if branch.virtual or branch.existence <= 0:
                continue
            positions = branch.sample(times)
            if trajectory_overlaps(states.T, positions, branch.headings(times),
                                   [self.config.ego_length, self.config.ego_width],
                                   branch.dimensions, self.config.safety_margin):
                raise PlanningFailure('Independent footprint check rejected the trajectory')
        self.warm_start = np.pad(coefficients, (0, 8 - len(coefficients)))
        return Plan(states.T, commands.T, accelerations.T, coefficients, elapsed,
                    stats.get('iter_count', 0), status,
                    dict(zip(self.component_names, np.asarray(self.objective_evaluate(coefficients, parameters)).ravel().tolist())))