import math

import casadi as ca
import numpy as np

from plan_predictive_cost_map.planner.config import VehicleConfig
from plan_predictive_cost_map.planner.dynamics import acceleration_command, dynamic_step
from plan_predictive_cost_map.pcm_lane_geometry import progress_reference_function, progress_reference_values


def polynomial_acceleration(elapsed, coefficients):
    return (coefficients[0] + coefficients[1] * elapsed
            + coefficients[2] * elapsed**2 + coefficients[3] * elapsed**3)


def acceleration_polynomial_step(time_step, settings=VehicleConfig(), clip_acceleration=True):
    state = ca.SX.sym('state', 7)
    coefficients = ca.SX.sym('acceleration_coefficients', 4)
    elapsed = ca.SX.sym('elapsed')
    steering = ca.SX.sym('steering')
    road = ca.SX.sym('road', 2)
    requested = polynomial_acceleration(elapsed, coefficients)
    next_state, acceleration = dynamic_step(time_step, settings, clip_acceleration)(
        state, ca.vertcat(requested, steering), road)
    return ca.Function('acceleration_polynomial_step',
                       [state, coefficients, elapsed, steering, road],
                       [next_state, acceleration, requested])


def build_kinematic_rollout(horizon, prediction_steps, command_time, settings=VehicleConfig(), reference_points=96):
    coefficients = ca.MX.sym('velocity_coefficients', 3)
    initial = ca.MX.sym('initial_state', 7)
    initial_ax = ca.MX.sym('initial_acceleration')
    initial_steering = ca.MX.sym('initial_steering')
    lane_y = ca.MX.sym('lane_y', 4)
    lane_z = ca.MX.sym('lane_z', 4)
    initial_pitch = ca.MX.sym('initial_pitch')
    reference = ca.MX.sym('progress_reference_data', 2 + 6 * reference_points)
    lookup = progress_reference_function(reference_points)
    times = ca.DM(np.linspace(horizon / prediction_steps, horizon, prediction_steps))
    speed = initial[4] + coefficients[0] * times + coefficients[1] * times**2 + coefficients[2] * times**3
    acceleration = coefficients[0] + 2.0 * coefficients[1] * times + 3.0 * coefficients[2] * times**2
    distance = (initial[4] * times + coefficients[0] * times**2 / 2.0
                + coefficients[1] * times**3 / 3.0 + coefficients[2] * times**4 / 4.0)
    geometry = ca.horzcat(*[lookup(distance[index], reference) for index in range(prediction_steps)]).T
    positions_x, positions_y, positions_z = geometry[:, 0], geometry[:, 1], geometry[:, 2]
    heading, pitch, curvature = geometry[:, 3], geometry[:, 4], geometry[:, 5]
    orientation = ca.horzcat(ca.sin(heading / 2.0) * ca.sin(pitch / 2.0),
                             -ca.cos(heading / 2.0) * ca.sin(pitch / 2.0),
                             ca.sin(heading / 2.0) * ca.cos(pitch / 2.0),
                             ca.cos(heading / 2.0) * ca.cos(pitch / 2.0))
    forward = ca.cos(initial_pitch) * positions_x - ca.sin(initial_pitch) * positions_z
    world_x = initial[0] + ca.cos(initial[3]) * forward - ca.sin(initial[3]) * positions_y
    world_y = initial[1] + ca.sin(initial[3]) * forward + ca.cos(initial[3]) * positions_y
    world_z = initial[2] + ca.sin(initial_pitch) * positions_x + ca.cos(initial_pitch) * positions_z
    yaw_rate = speed * curvature
    states = ca.horzcat(world_x, world_y, world_z, initial[3] + heading, speed,
                        ca.MX.zeros(prediction_steps, 1), yaw_rate)
    command_speed = (initial[4] + coefficients[0] * command_time
                      + coefficients[1] * command_time**2 + coefficients[2] * command_time**3)
    return ca.Function('longitudinal_kinematic_rollout',
                       [coefficients, initial, initial_ax, initial_steering, lane_y, lane_z, initial_pitch, reference],
                       [ca.horzcat(positions_x, positions_y, positions_z), speed,
                        ca.horzcat(acceleration, speed**2 * curvature, ca.MX.zeros(prediction_steps, 1)),
                        acceleration, command_speed, ca.atan(settings.wheelbase * curvature),
                        ca.MX.zeros(0, 1), orientation, states, distance])


def dynamic_light_sample_times(horizon, prediction_steps, control_step, command_time):
    prediction_times = np.linspace(horizon / prediction_steps, horizon, prediction_steps)
    boundaries = np.unique(np.round(np.concatenate((np.arange(0.0, horizon, control_step),
                                                    prediction_times, [0.0, command_time, horizon])), 12))
    return np.sort(np.r_[boundaries, 0.5 * (boundaries[:-1] + boundaries[1:])])


def velocity_polynomial(elapsed, coefficients, initial_speed, horizon=1.0):
    normalized = elapsed / horizon
    speed = initial_speed + sum(coefficients[index] * normalized**(index + 1) for index in range(4))
    acceleration = sum((index + 1) * coefficients[index] * normalized**index / horizon for index in range(4))
    progress = initial_speed * elapsed + horizon * sum(coefficients[index] * normalized**(index + 2)
                                                        / (index + 2) for index in range(4))
    return speed, acceleration, progress


def shift_velocity_coefficients(coefficients, delta, horizon):
    # Exact re-expansion of v(t + delta) - v(delta); the new measured speed supplies the constant term.
    shift = delta / horizon
    shifted = np.zeros(len(coefficients) + 1)
    for power, value in enumerate(np.r_[0.0, np.asarray(coefficients, dtype=float).flatten()]):
        for order in range(power + 1):
            shifted[order] += value * math.comb(power, order) * shift**(power - order)
    return shifted[1:]


def velocity_profile_coefficients(horizon, initial_speed, target_speed, samples=80, degree=4, normalized=True):
    times = np.linspace(horizon / samples, horizon, samples)
    basis = np.column_stack([((times / horizon) if normalized else times)**(power + 1) for power in range(degree)])
    profiles = [np.full(samples, initial_speed)]
    profiles += [np.maximum(0.0, initial_speed - rate * times) for rate in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)]
    ceiling = max(target_speed, initial_speed)
    profiles += [np.minimum(ceiling, initial_speed + rate * times) for rate in (0.5, 1.0, 2.0)]
    return [np.linalg.lstsq(basis, profile - initial_speed, rcond=None)[0] for profile in profiles]


def build_dynamic_light_rollout(horizon, prediction_steps, control_step, command_time,
                                settings=VehicleConfig(), use_road_gradient=True, road_pitch=0.0,
                                reference_points=96):
    coefficients = ca.MX.sym('velocity_coefficients', 4)
    initial = ca.MX.sym('initial_state', 7)
    initial_ax = ca.MX.sym('initial_acceleration')
    initial_steering = ca.MX.sym('initial_steering')
    lane_y = ca.MX.sym('lane_y', 4)
    lane_z = ca.MX.sym('lane_z', 4)
    initial_pitch = ca.MX.sym('initial_pitch')
    reference = ca.MX.sym('progress_reference_data', 2 + 6 * reference_points)

    times = dynamic_light_sample_times(horizon, prediction_steps, control_step, command_time)
    dense_speed, dense_acceleration, dense_progress = velocity_polynomial(ca.DM(times), coefficients, initial[4], horizon)
    dense_angles = progress_reference_values(dense_progress, reference, reference_points, channels=(3, 4))
    tangent_x = ca.cos(dense_angles[:, 1]) * ca.cos(dense_angles[:, 0])
    tangent_y = ca.cos(dense_angles[:, 1]) * ca.sin(dense_angles[:, 0])
    tangent_z = ca.sin(dense_angles[:, 1])
    vertical = ca.sin(initial_pitch) * tangent_x + ca.cos(initial_pitch) * tangent_z
    horizontal = ca.sqrt(ca.fmax((ca.cos(initial_pitch) * tangent_x - ca.sin(initial_pitch) * tangent_z)**2
                                 + tangent_y**2, 1e-12))
    grade = ca.atan2(vertical, horizontal) if use_road_gradient else road_pitch
    resistance = (0.5 * settings.air_density * settings.drag_coefficient * settings.frontal_area
                  * dense_speed**2 / settings.mass + settings.viscous_resistance * dense_speed / settings.mass
                  + settings.rolling_resistance_coefficient * 9.81 * ca.cos(grade) + 9.81 * ca.sin(grade))
    dense_requests = dense_acceleration + resistance
    prediction_times = np.round(np.linspace(horizon / prediction_steps, horizon, prediction_steps), 12)
    indices = [int(np.argmin(np.abs(times - elapsed))) for elapsed in prediction_times]
    command_index = int(np.argmin(np.abs(times - command_time)))
    progress, speed = dense_progress[indices], dense_speed[indices]
    geometry = progress_reference_values(progress, reference, reference_points)
    positions = geometry[:, :3]
    heading, pitch, curvature = geometry[:, 3], geometry[:, 4], geometry[:, 5]
    orientation = ca.horzcat(ca.sin(heading / 2.0) * ca.sin(pitch / 2.0),
                             -ca.cos(heading / 2.0) * ca.sin(pitch / 2.0),
                             ca.sin(heading / 2.0) * ca.cos(pitch / 2.0),
                             ca.cos(heading / 2.0) * ca.cos(pitch / 2.0))
    forward = ca.cos(initial_pitch) * positions[:, 0] - ca.sin(initial_pitch) * positions[:, 2]
    world_x = initial[0] + ca.cos(initial[3]) * forward - ca.sin(initial[3]) * positions[:, 1]
    world_y = initial[1] + ca.sin(initial[3]) * forward + ca.cos(initial[3]) * positions[:, 1]
    world_z = initial[2] + ca.sin(initial_pitch) * positions[:, 0] + ca.cos(initial_pitch) * positions[:, 2]
    states = ca.horzcat(world_x, world_y, world_z, initial[3] + heading, speed,
                        ca.MX.zeros(prediction_steps, 1), speed * curvature)
    steering = ca.fmin(settings.maximum_steering,
                       ca.fmax(-settings.maximum_steering, ca.atan(settings.wheelbase * curvature)))
    return ca.Function('longitudinal_dynamic_light_rollout',
        [coefficients, initial, initial_ax, initial_steering, lane_y, lane_z, initial_pitch, reference],
        [positions, speed, ca.horzcat(dense_acceleration[indices], speed**2 * curvature,
                                      ca.MX.zeros(prediction_steps, 1)),
         dense_requests[indices], dense_speed[command_index], steering,
         dense_requests, orientation, states, progress])


def build_dynamic_rollout(horizon, prediction_steps, control_step, command_time,
                          settings=VehicleConfig(), use_road_gradient=True, road_pitch=0.0, reference_points=96,
                          powertrain=None, efficiency=None, actuator_vehicle_model='dynamic',
                          actuator_method='detailed', actuator_stage_step=0.005):
    coefficients = ca.MX.sym('acceleration_coefficients', 4)
    initial = ca.MX.sym('initial_state', 7)
    initial_ax = ca.MX.sym('initial_acceleration')
    initial_steering = ca.MX.sym('initial_steering')
    lane_y = ca.MX.sym('lane_y', 4)
    lane_z = ca.MX.sym('lane_z', 4)
    initial_pitch = ca.MX.sym('initial_pitch')
    reference = ca.MX.sym('progress_reference_data', 2 + 6 * reference_points)
    actuator_initial = ca.MX.sym('actuator_initial', 4) if powertrain is not None else None
    lookup = progress_reference_function(reference_points)
    cosine_yaw, sine_yaw = ca.cos(initial[3]), ca.sin(initial[3])
    cosine_pitch, sine_pitch = ca.cos(initial_pitch), ca.sin(initial_pitch)

    def local_position(state):
        delta_x, delta_y, delta_z = state[0] - initial[0], state[1] - initial[1], state[2] - initial[2]
        forward = cosine_yaw * delta_x + sine_yaw * delta_y
        return ca.vertcat(cosine_pitch * forward + sine_pitch * delta_z,
                          -sine_yaw * delta_x + cosine_yaw * delta_y,
                          -sine_pitch * forward + cosine_pitch * delta_z)

    def road_and_steering(state, progress):
        geometry = lookup(progress, reference)
        tangent_x = ca.cos(geometry[4]) * ca.cos(geometry[3])
        tangent_y = ca.cos(geometry[4]) * ca.sin(geometry[3])
        tangent_z = ca.sin(geometry[4])
        vertical = sine_pitch * tangent_x + cosine_pitch * tangent_z
        horizontal = ca.sqrt(ca.fmax((cosine_pitch * tangent_x - sine_pitch * tangent_z)**2 + tangent_y**2, 1e-12))
        pitch = ca.atan2(vertical, horizontal)
        lookahead = ca.fmax(settings.minimum_lookahead, settings.lookahead_time * state[4])
        target = lookup(progress + lookahead, reference)
        target_x, target_y, target_z = target[0], target[1], target[2]
        forward = cosine_pitch * target_x - sine_pitch * target_z
        target_map_x = initial[0] + cosine_yaw * forward - sine_yaw * target_y
        target_map_y = initial[1] + sine_yaw * forward + cosine_yaw * target_y
        target_map_z = initial[2] + sine_pitch * target_x + cosine_pitch * target_z
        delta_x, delta_y = target_map_x - state[0], target_map_y - state[1]
        target_forward = (ca.cos(pitch) * (ca.cos(state[3]) * delta_x + ca.sin(state[3]) * delta_y)
                          + ca.sin(pitch) * (target_map_z - state[2]))
        target_lateral = -ca.sin(state[3]) * delta_x + ca.cos(state[3]) * delta_y
        squared_distance = target_forward**2 + target_lateral**2
        steering = ca.if_else(squared_distance < 1e-8, 0.0,
                              ca.atan2(2.0 * settings.wheelbase * target_lateral,
                                       ca.fmax(squared_distance, 1e-8)))
        steering = ca.fmin(settings.maximum_steering, ca.fmax(-settings.maximum_steering, steering))
        load_pitch = pitch if use_road_gradient else road_pitch
        return ca.vertcat(pitch, load_pitch), steering

    def quaternion(yaw, pitch):
        return ca.vertcat(ca.sin(yaw / 2.0) * ca.sin(pitch / 2.0),
                          -ca.cos(yaw / 2.0) * ca.sin(pitch / 2.0),
                          ca.sin(yaw / 2.0) * ca.cos(pitch / 2.0),
                          ca.cos(yaw / 2.0) * ca.cos(pitch / 2.0))

    initial_rotation = quaternion(initial[3], initial_pitch)

    initial_road, _ = road_and_steering(initial, 0.0)
    if powertrain is None:
        initial_request = acceleration_command(settings)(initial, initial_ax, initial_steering, initial_road[1])
        polynomial = ca.vertcat(initial_request + coefficients[0], coefficients[1:4])
    else:
        from plan_predictive_cost_map.pcm_powertrain import build_actuator_interval
        joint = ca.vertcat(initial, actuator_initial, ca.MX.zeros(4, 1))
    prediction_times = np.round(np.linspace(horizon / prediction_steps, horizon, prediction_steps), 12)
    boundaries = np.unique(np.round(np.concatenate((np.arange(0.0, horizon, control_step),
                                                    prediction_times, [0.0, command_time, horizon])), 12))
    state = initial
    positions, speeds, accelerations, requests, steering_samples, input_samples = [], [], [], [], [], []
    orientations, states = [], []
    progress = ca.MX(0.0)
    progress_samples = []
    actuator_samples, command_samples = [], []
    steps = {}
    command_speed = initial[4]
    for previous_time, elapsed in zip(boundaries[:-1], boundaries[1:]):
        duration = round(float(elapsed - previous_time), 12)
        if duration not in steps:
            steps[duration] = (acceleration_polynomial_step(duration, settings, clip_acceleration=False)
                if powertrain is None else build_actuator_interval(duration, horizon, command_time,
                    efficiency, powertrain, settings, actuator_vehicle_model, actuator_method, actuator_stage_step))
        road, steering = road_and_steering(state, progress)
        previous_speed = state[4]
        if powertrain is None:
            state, acceleration, requested = steps[duration](state, polynomial, previous_time, steering, road)
            progress += 0.5 * (previous_speed + state[4]) * duration
        else:
            joint, acceleration, requested = steps[duration](joint, coefficients, initial[4], road, steering)
            state, progress = joint[:7], joint[13]
            command_samples.append(velocity_polynomial(min(previous_time + command_time, horizon),
                                                       coefficients, initial[4], horizon)[0])
        input_samples.append(requested)
        if elapsed == round(command_time, 12):
            command_speed = (state[4] if powertrain is None else
                velocity_polynomial(command_time, coefficients, initial[4], horizon)[0])
        if elapsed in prediction_times:
            positions.append(local_position(state).T)
            speeds.append(state[4])
            accelerations.append(acceleration.T)
            requests.append(polynomial_acceleration(elapsed, polynomial) if powertrain is None else requested)
            steering_samples.append(steering)
            physical_pitch = road_and_steering(state, progress)[0][0]
            rotation = quaternion(state[3], physical_pitch)
            relative_vector = (initial_rotation[3] * rotation[:3] - rotation[3] * initial_rotation[:3]
                                - ca.cross(initial_rotation[:3], rotation[:3]))
            relative_scalar = initial_rotation[3] * rotation[3] + ca.dot(initial_rotation[:3], rotation[:3])
            orientations.append(ca.vertcat(relative_vector, relative_scalar).T)
            states.append(state.T)
            progress_samples.append(progress)
            if powertrain is not None:
                actuator_samples.append(joint[7:13].T)
    positions = ca.vertcat(*positions)
    reference_height = ca.vertcat(*[lookup(value, reference)[2] for value in progress_samples])
    positions = ca.horzcat(positions[:, :2], reference_height)
    inputs = [coefficients, initial, initial_ax, initial_steering, lane_y, lane_z, initial_pitch, reference]
    outputs = [positions, ca.vertcat(*speeds), ca.vertcat(*accelerations),
               ca.vertcat(*requests), command_speed, ca.vertcat(*steering_samples),
               ca.vertcat(*input_samples, polynomial_acceleration(horizon, polynomial))
                    if powertrain is None else ca.vertcat(*input_samples, requested),
               ca.vertcat(*orientations), ca.vertcat(*states), ca.vertcat(*progress_samples)]
    if powertrain is not None:
        if command_time == 0.0:
            outputs[4] = initial[4]
        inputs.append(actuator_initial)
        outputs.extend([joint[11], joint[12], joint[11] - joint[12], ca.vertcat(*actuator_samples),
                        ca.vertcat(*command_samples, velocity_polynomial(horizon, coefficients, initial[4], horizon)[0])])
    return ca.Function('longitudinal_dynamic_rollout' if powertrain is None else 'longitudinal_dynamic_actuators_rollout',
                       inputs, outputs)