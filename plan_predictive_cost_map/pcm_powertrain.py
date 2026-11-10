import csv
from dataclasses import dataclass, fields, replace
import math
from pathlib import Path
import re

import casadi as ca
import numpy as np
from scipy.linalg import expm

from plan_predictive_cost_map.planner.config import VehicleConfig


@dataclass(frozen=True)
class PowertrainConfig:
    motor_gear_ratio: float = 9.0
    wheel_gear_ratio: float = 1.0
    wheel_radius: float = 0.32
    motor_inertia: float = 0.02
    wheel_inertia: float = 1.2
    motor_viscous_friction: float = 0.001
    wheel_viscous_friction: float = 0.002
    shaft_stiffness: float = 18000.0
    shaft_damping: float = 120.0
    maximum_motor_torque: float = 200.0
    maximum_regenerative_torque: float = 120.0
    motor_torque_rate_limit: float = 1000.0
    integration_step: float = 0.001
    acceleration_command_time_constant: float = 0.05
    speed_gain: float = 6.0

    def __post_init__(self):
        nonnegative = {'motor_viscous_friction', 'wheel_viscous_friction',
                       'shaft_damping', 'acceleration_command_time_constant'}
        for setting in fields(self):
            value = getattr(self, setting.name)
            if not math.isfinite(value) or value < 0.0 or (value == 0.0 and setting.name not in nonnegative):
                raise ValueError('Invalid powertrain setting: ' + setting.name)


def motor_efficiency_function(path, model='lut', degree=4, tolerance=0.05, diagnostics=None):
    if (model not in ('lut', 'polynomial', 'spline') or not isinstance(degree, int) or not 1 <= degree <= 8
            or not math.isfinite(tolerance) or tolerance <= 0.0):
        raise ValueError('Efficiency model must be lut, spline or polynomial (degree 1..8) with positive tolerance')
    with Path(path).open(newline='') as stream:
        rows = list(csv.reader(stream))
    if not rows or rows[0][:2] != ['mode', 'speed_rpm'] or len(rows[0]) < 5:
        raise ValueError('Invalid motor efficiency map header')
    torque_matches = [re.fullmatch(r'torque_([0-9]+(?:\.[0-9]+)?)_nm', name) for name in rows[0][2:]]
    if not all(torque_matches):
        raise ValueError('Invalid motor efficiency torque axis')
    torques = np.asarray([float(match.group(1)) for match in torque_matches])
    tables = {'motoring': [], 'regeneration': []}
    speeds = {'motoring': [], 'regeneration': []}
    for row in rows[1:]:
        if not row:
            continue
        if len(row) != len(rows[0]) or row[0] not in tables:
            raise ValueError('Invalid motor efficiency map row')
        speeds[row[0]].append(float(row[1]))
        tables[row[0]].append([float(value) for value in row[2:]])
    speed_grid = np.asarray(speeds['motoring'])
    if (len(speed_grid) < 2 or speeds['motoring'] != speeds['regeneration']
            or not np.isfinite(speed_grid).all() or np.any(speed_grid < 0.0)
            or np.any(np.diff(speed_grid) <= 0.0) or not np.isfinite(torques).all()
            or np.any(torques < 0.0) or np.any(np.diff(torques) <= 0.0)):
        raise ValueError('Efficiency map axes must be finite, increasing, and match between modes')
    motor_speed = ca.SX.sym('motor_speed')
    motor_torque = ca.SX.sym('motor_torque')
    query = ca.vertcat(ca.fmin(speed_grid[-1], ca.fmax(speed_grid[0], ca.fabs(motor_speed) * 60.0 / (2.0 * math.pi))),
                      ca.fmin(torques[-1], ca.fmax(torques[0], ca.fabs(motor_torque))))
    values = []
    dense_speed = np.linspace(speed_grid[0], speed_grid[-1], 241)
    dense_torque = np.linspace(torques[0], torques[-1], 201)
    if model == 'spline':
        speed_basis = _cubic_spline_basis(dense_speed, speed_grid, np.maximum)
        torque_basis = _cubic_spline_basis(dense_torque, torques, np.maximum)
        query_basis = (_cubic_spline_basis(query[0], speed_grid, ca.fmax),
                       _cubic_spline_basis(query[1], torques, ca.fmax))
    for mode, table in tables.items():
        table = np.asarray(table)
        if (table.shape != (len(speed_grid), len(torques)) or not np.isfinite(table).all()
                or np.any(table <= 0.0) or np.any(table > 1.0)):
            raise ValueError('Motor efficiencies must be in (0, 1] and match the map axes')
        lookup = ca.interpolant(mode + '_efficiency', 'linear',
                                [speed_grid.tolist(), torques.tolist()], table.flatten(order='F'))
        if model == 'lut':
            values.append(lookup(query))
            continue
        # Least-squares fit to the densely resampled bilinear map, so the surrogate tracks the simulator LUT.
        grid_speed, grid_torque = np.meshgrid(dense_speed, dense_torque, indexing='ij')
        target = lookup(np.stack([grid_speed.ravel(), grid_torque.ravel()])).full().reshape(grid_speed.shape)
        if model == 'spline':
            weights = np.linalg.pinv(speed_basis) @ target @ np.linalg.pinv(torque_basis).T
            full_error = np.max(np.abs(speed_basis @ weights @ torque_basis.T - target))
            # Separable low-rank form sum_r f_r(speed) g_r(torque); SVD in orthonormalized bases.
            speed_q, speed_r = np.linalg.qr(speed_basis)
            torque_q, torque_r = np.linalg.qr(torque_basis)
            left, singular, right = np.linalg.svd(speed_r @ weights @ torque_r.T)
            for rank in range(1, len(singular) + 1):
                speed_factors = np.linalg.solve(speed_r, left[:, :rank] * singular[:rank])
                torque_factors = np.linalg.solve(torque_r, right[:rank].T)
                fitted = speed_basis @ speed_factors @ (torque_basis @ torque_factors).T
                if np.max(np.abs(fitted - target)) <= 1.1 * full_error + 1e-6:
                    break
            values.append(ca.dot(ca.mtimes(ca.DM(speed_factors).T, query_basis[0]),
                                 ca.mtimes(ca.DM(torque_factors).T, query_basis[1])))
        else:
            x, y = grid_speed / speed_grid[-1], grid_torque / torques[-1]
            terms = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
            basis = np.stack([(x**i * y**j).ravel() for i, j in terms], axis=1)
            weights = np.linalg.lstsq(basis, target.ravel(), rcond=None)[0]
            fitted = (basis @ weights).reshape(target.shape)
            values.append(sum(float(weight) * (query[0] / speed_grid[-1])**i * (query[1] / torques[-1])**j
                              for weight, (i, j) in zip(weights, terms)))
        error = np.abs(fitted - target)
        if diagnostics is not None:
            diagnostics[mode] = {'max_error': float(error.max()), 'rms_error': float(np.sqrt(np.mean(error**2))),
                                 'rank': rank if model == 'spline' else None}
        if np.min(fitted) <= 0.05 or np.max(fitted) > 1.0 or error.max() > tolerance:
            raise ValueError(f'{model} {mode} efficiency fit has max error {error.max():.4f} (tolerance {tolerance}) '
                             f'or leaves (0.05, 1]; use the lut model')
    return ca.Function('motor_efficiency', [motor_speed, motor_torque], values)


def _cubic_spline_basis(value, knots, maximum):
    # C2 cubic regression-spline basis (truncated powers) with interior knots on the map axis.
    scaled = (value - knots[0]) / (knots[-1] - knots[0])
    square = scaled * scaled
    columns = [scaled * 0.0 + 1.0, scaled, square, square * scaled]
    for knot in (knots[1:-1] - knots[0]) / (knots[-1] - knots[0]):
        shifted = maximum(0.0, scaled - knot)
        columns.append(shifted * shifted * shifted)
    return np.stack(columns, axis=-1) if isinstance(value, np.ndarray) else ca.vertcat(*columns)


def build_powertrain_step(time_step, efficiency, settings=PowertrainConfig(), vehicle=VehicleConfig()):
    state = ca.SX.sym('powertrain_state', 6)
    speed = ca.SX.sym('vehicle_speed')
    request = ca.SX.sym('acceleration_request')
    motor_speed, shaft_twist, motor_torque, filtered, drawn, recovered = ca.vertsplit(state)
    if not math.isfinite(time_step) or time_step <= 0.0:
        raise ValueError('Powertrain time step must be finite and positive')
    count = math.ceil(time_step / settings.integration_step)
    if count > 100000:
        raise ValueError('Powertrain update requires too many substeps')
    duration = time_step / count
    inertia = settings.motor_inertia * settings.motor_gear_ratio + settings.wheel_inertia / settings.motor_gear_ratio
    friction = settings.motor_gear_ratio * settings.motor_viscous_friction + settings.wheel_viscous_friction / settings.motor_gear_ratio
    ratio = settings.motor_gear_ratio * settings.wheel_gear_ratio / settings.wheel_radius
    filter_fraction = (-math.expm1(-duration / settings.acceleration_command_time_constant)
                       if settings.acceleration_command_time_constant > 0.0 else 1.0)
    force_integral = 0.0
    electrical_power = 0.0
    motor_efficiency = 0.0
    for _ in range(count):
        filtered += filter_fraction * (request - filtered)
        bounded = ca.fmin(vehicle.maximum_acceleration, ca.fmax(-vehicle.maximum_deceleration, filtered))
        requested_twist = vehicle.mass * bounded * settings.wheel_radius / (settings.shaft_stiffness * settings.wheel_gear_ratio)
        requested_torque = (inertia * ratio * bounded + friction * motor_speed +
                            settings.shaft_stiffness * requested_twist) / settings.motor_gear_ratio
        bounded_torque = ca.fmin(settings.maximum_motor_torque, ca.fmax(-settings.maximum_regenerative_torque, requested_torque))
        torque_change = settings.motor_torque_rate_limit * duration
        motor_torque += ca.fmin(torque_change, ca.fmax(-torque_change, bounded_torque - motor_torque))
        mechanical_power = motor_torque * motor_speed
        motoring_efficiency, regeneration_efficiency = efficiency(motor_speed, motor_torque)
        motor_efficiency = ca.if_else(mechanical_power < 0.0, regeneration_efficiency, motoring_efficiency)
        electrical_power = ca.if_else(mechanical_power < 0.0,
                                     mechanical_power * motor_efficiency, mechanical_power / motor_efficiency)
        drawn += ca.fmax(0.0, electrical_power) * duration
        recovered += ca.fmax(0.0, -electrical_power) * duration
        twist_rate = motor_speed / settings.motor_gear_ratio - settings.wheel_gear_ratio * speed / settings.wheel_radius
        shaft_torque = settings.shaft_stiffness * shaft_twist + settings.shaft_damping * twist_rate
        wheel_force = settings.wheel_gear_ratio * shaft_torque / settings.wheel_radius
        wheel_force += ca.if_else(bounded < 0.0, ca.fmin(0.0, vehicle.mass * bounded - wheel_force), 0.0)
        force_integral += wheel_force * duration
        motor_speed = ca.fmax(0.0, motor_speed + duration *
            (-friction * motor_speed - shaft_torque + settings.motor_gear_ratio * motor_torque) / inertia)
        shaft_twist += duration * twist_rate
    next_state = ca.vertcat(motor_speed, shaft_twist, motor_torque, filtered, drawn, recovered)
    return ca.Function('powertrain_step', [state, speed, request],
                       [next_state, force_integral / time_step, electrical_power, motor_efficiency])


def build_actuator_interval(time_step, horizon, command_time, efficiency, settings=PowertrainConfig(),
                            vehicle=VehicleConfig(), vehicle_model='dynamic', method='detailed', stage_step=0.005,
                            held_speed_command=False):
    from plan_predictive_cost_map.planner.dynamics import dynamic_step

    if vehicle_model not in ('kinematic', 'dynamic'):
        raise ValueError('Actuator vehicle model must be kinematic or dynamic')
    if method not in ('detailed', 'discrete') or not math.isfinite(stage_step) or stage_step <= 0.0:
        raise ValueError('Invalid actuator integration method or stage step')
    count = math.ceil(time_step / (min(settings.integration_step, vehicle.integration_step)
                                   if method == 'detailed' else stage_step))
    if count < 1 or count > 100000:
        raise ValueError('Invalid coupled powertrain integration interval')
    duration = time_step / count
    joint = ca.SX.sym('actuated_state', 15)
    coefficients = ca.SX.sym('speed_command_coefficients', 1 if held_speed_command else 4)
    base_speed = ca.SX.sym('initial_speed')
    road = ca.SX.sym('road', 2)
    steering = ca.SX.sym('steering')
    command_elapsed = ca.SX.sym('command_elapsed')
    normalized = ca.fmin(horizon, command_elapsed + command_time) / horizon
    target = (coefficients[0] if held_speed_command else
              base_speed + sum(coefficients[index] * normalized**(index + 1) for index in range(4)))
    request = ca.if_else(target <= 1e-6, -vehicle.maximum_deceleration,
        ca.fmin(vehicle.maximum_acceleration,
                ca.fmax(-vehicle.maximum_deceleration, settings.speed_gain * (target - joint[4]))))
    if method == 'detailed':
        actuator_step = build_powertrain_step(duration, efficiency, settings, vehicle)
        powertrain, force, _, _ = actuator_step(joint[7:13], joint[4], request)
        stage_vehicle = vehicle
    else:
        discrete = build_discrete_powertrain_step(duration, efficiency, settings, vehicle, vehicle_model)
        powertrain, next_speed = discrete(joint, request, steering, road[1])
        stage_vehicle = replace(vehicle, integration_step=duration)
        aerodynamic = 0.5 * vehicle.air_density * vehicle.drag_coefficient * vehicle.frontal_area * joint[4]**2 / vehicle.mass
        loads = (aerodynamic + vehicle.viscous_resistance * joint[4] / vehicle.mass +
                 9.81 * ca.sin(road[1]) + vehicle.rolling_resistance_coefficient * 9.81 * ca.cos(road[1]))
        fraction = ca.fmin(1.0, ca.fmax(0.0, (joint[4] - vehicle.kinematic_blend_speed) /
                                       (vehicle.dynamic_blend_speed - vehicle.kinematic_blend_speed)))
        weight = fraction**2 * (3.0 - 2.0 * fraction)
        front_force = vehicle.front_cornering_stiffness * (steering -
            (joint[5] + vehicle.front_axle_distance * joint[6]) / ca.fmax(joint[4], vehicle.kinematic_blend_speed))
        coupling = weight * (joint[6] * joint[5] - front_force * ca.sin(steering) / vehicle.mass)
        force = vehicle.mass * ((next_speed - joint[4]) / duration + loads -
                                (coupling if vehicle_model == 'dynamic' else 0.0))
    if vehicle_model == 'dynamic':
        next_vehicle, acceleration = dynamic_step(duration, stage_vehicle, clip_acceleration=method == 'detailed')(
            joint[:7], ca.vertcat(force / vehicle.mass, steering), road)
    else:
        loads = ((0.5 * vehicle.air_density * vehicle.drag_coefficient * vehicle.frontal_area * joint[4]**2 +
              vehicle.viscous_resistance * joint[4]) / vehicle.mass +
             vehicle.rolling_resistance_coefficient * 9.81 * ca.cos(road[1]) + 9.81 * ca.sin(road[1]))
        drive = ca.fmin(vehicle.maximum_acceleration,
                       ca.fmax(-vehicle.maximum_deceleration, force / vehicle.mass - loads))
        speed = ca.fmax(0.0, joint[4] + duration * drive)
        yaw_rate = ca.tan(steering) * speed / vehicle.wheelbase
        midpoint_yaw = joint[3] + 0.5 * yaw_rate * duration
        yaw = joint[3] + yaw_rate * duration
        next_vehicle = ca.vertcat(joint[0] + duration * speed * ca.cos(road[0]) * ca.cos(midpoint_yaw),
                                  joint[1] + duration * speed * ca.cos(road[0]) * ca.sin(midpoint_yaw),
                                  joint[2] + duration * speed * ca.sin(road[0]),
                                  ca.atan2(ca.sin(yaw), ca.cos(yaw)), speed, 0.0, yaw_rate)
        acceleration = ca.vertcat((speed - joint[4]) / duration, speed * yaw_rate,
                                  (yaw_rate - joint[6]) / duration)
    progress = joint[13] + 0.5 * (joint[4] + next_vehicle[4]) * duration
    next_joint = ca.vertcat(next_vehicle, powertrain, progress, joint[14] + duration)
    step = ca.Function('actuated_vehicle_microstep', [joint, coefficients, base_speed, road, steering, command_elapsed],
                       [next_joint, acceleration, request])
    initial = ca.MX.sym('actuated_initial', 15)
    commands = ca.MX.sym('actuated_commands', 1 if held_speed_command else 4)
    speed_initial = ca.MX.sym('base_speed')
    road_input = ca.MX.sym('road_input', 2)
    steering_input = ca.MX.sym('steering_input')
    history, acceleration_history, requests = step.mapaccum(count)(
        initial, ca.repmat(commands, 1, count), ca.repmat(speed_initial, 1, count),
        ca.repmat(road_input, 1, count), ca.repmat(steering_input, 1, count),
        ca.repmat(initial[14], 1, count))
    final = history[:, -1]
    averaged = ca.vertcat((final[4] - initial[4]) / time_step,
                          ca.sum2(acceleration_history[1, :]) / count,
                          (final[6] - initial[6]) / time_step)
    return ca.Function('actuated_vehicle_interval', [initial, commands, speed_initial, road_input, steering_input],
                       [final, averaged, requests[0, -1]])


def build_discrete_powertrain_step(time_step, efficiency, settings=PowertrainConfig(),
                                   vehicle=VehicleConfig(), vehicle_model='dynamic'):
    joint = ca.SX.sym('discrete_joint', 15)
    request = ca.SX.sym('discrete_request')
    steering = ca.SX.sym('discrete_steering')
    pitch = ca.SX.sym('discrete_pitch')
    gear, wheel_gear, radius = settings.motor_gear_ratio, settings.wheel_gear_ratio, settings.wheel_radius
    inertia = settings.motor_inertia * gear + settings.wheel_inertia / gear
    friction = gear * settings.motor_viscous_friction + settings.wheel_viscous_friction / gear
    stiffness, damping = settings.shaft_stiffness, settings.shaft_damping
    system = np.array([[-(friction + damping / gear) / inertia, -stiffness / inertia,
                        damping * wheel_gear / (radius * inertia)],
                       [1.0 / gear, 0.0, -wheel_gear / radius],
                       [damping * wheel_gear / (gear * radius * vehicle.mass),
                        stiffness * wheel_gear / (radius * vehicle.mass),
                        -damping * wheel_gear**2 / (radius**2 * vehicle.mass)]])
    forcing = np.array([[gear / inertia, 0.0], [0.0, 0.0], [0.0, 1.0]])
    augmented = np.zeros((5, 5))
    augmented[:3, :3], augmented[:3, 3:] = system, forcing
    def transition(fraction):
        matrix = expm(augmented * time_step * fraction)
        return ca.DM(matrix[:3, :3]), ca.DM(matrix[:3, 3:])
    def filtered_at(fraction):
        decay = (math.exp(-time_step * fraction / settings.acceleration_command_time_constant)
                 if settings.acceleration_command_time_constant > 0.0 else 0.0)
        return request + decay * (joint[10] - request)
    filtered = filtered_at(1.0)
    bounded = ca.fmin(vehicle.maximum_acceleration, ca.fmax(-vehicle.maximum_deceleration, filtered_at(0.5)))
    requested_torque = ((inertia * gear * wheel_gear / radius + vehicle.mass * radius / wheel_gear) * bounded +
                        friction * joint[7]) / gear
    target_torque = ca.fmin(settings.maximum_motor_torque, ca.fmax(-settings.maximum_regenerative_torque, requested_torque))
    change_limit = settings.motor_torque_rate_limit * time_step
    torque_change = ca.fmin(change_limit, ca.fmax(-change_limit, target_torque - joint[9]))
    torque = joint[9] + torque_change
    ramp_time = ca.fabs(torque_change) / settings.motor_torque_rate_limit
    mean_torque = torque - torque_change * ramp_time / (2.0 * time_step)
    shaft_force = wheel_gear / radius * (stiffness * joint[8] + damping * (joint[7] / gear - wheel_gear * joint[4] / radius))
    brake = ca.if_else(bounded < 0.0, ca.fmin(0.0, bounded - shaft_force / vehicle.mass), 0.0)
    force_clamp = ca.fmin(vehicle.maximum_acceleration,
                         ca.fmax(-vehicle.maximum_deceleration, shaft_force / vehicle.mass + brake))
    loads = ((0.5 * vehicle.air_density * vehicle.drag_coefficient * vehicle.frontal_area * joint[4]**2 +
              vehicle.viscous_resistance * joint[4]) / vehicle.mass +
             vehicle.rolling_resistance_coefficient * 9.81 * ca.cos(pitch) + 9.81 * ca.sin(pitch))
    fraction = ca.fmin(1.0, ca.fmax(0.0, (joint[4] - vehicle.kinematic_blend_speed) /
                                   (vehicle.dynamic_blend_speed - vehicle.kinematic_blend_speed)))
    weight = fraction**2 * (3.0 - 2.0 * fraction)
    front_force = vehicle.front_cornering_stiffness * (steering -
        (joint[5] + vehicle.front_axle_distance * joint[6]) / ca.fmax(joint[4], vehicle.kinematic_blend_speed))
    lateral = weight * (joint[6] * joint[5] - front_force * ca.sin(steering) / vehicle.mass)
    constant = force_clamp - shaft_force / vehicle.mass - loads + (lateral if vehicle_model == 'dynamic' else 0.0)
    initial = ca.vertcat(joint[7], joint[8], joint[4])
    inputs = ca.vertcat(mean_torque, constant)
    state_matrix, input_matrix = transition(1.0)
    final = state_matrix @ initial + input_matrix @ inputs
    drawn, recovered = joint[11], joint[12]
    for location in (0.5 - math.sqrt(3.0) / 6.0, 0.5 + math.sqrt(3.0) / 6.0):
        state_matrix, input_matrix = transition(location)
        sample = state_matrix @ initial + input_matrix @ inputs
        ramp_fraction = ca.fmin(1.0, location * time_step / ca.fmax(ramp_time, 1e-12))
        sample_torque = joint[9] + torque_change * ramp_fraction
        mechanical = ca.fmax(0.0, sample[0]) * sample_torque
        motoring, regeneration = efficiency(sample[0], sample_torque)
        power = ca.if_else(mechanical < 0.0, mechanical * regeneration, mechanical / motoring)
        drawn += ca.fmax(0.0, power) * time_step / 2.0
        recovered += ca.fmax(0.0, -power) * time_step / 2.0
    powertrain = ca.vertcat(ca.fmax(0.0, final[0]), final[1], torque, filtered, drawn, recovered)
    return ca.Function('discrete_powertrain_step', [joint, request, steering, pitch],
                       [powertrain, ca.fmax(0.0, final[2])])