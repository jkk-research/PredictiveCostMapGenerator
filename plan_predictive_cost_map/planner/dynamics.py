"""CasADi counterpart of simulator/src/vehicle_model.cpp's dynamic update."""

import math
import casadi as ca

from .config import VehicleConfig


def acceleration_command(settings=VehicleConfig()):
    state = ca.SX.sym('state', 7)
    desired = ca.SX.sym('desired')
    steering = ca.SX.sym('steering')
    pitch = ca.SX.sym('pitch')
    speed, lateral, yaw_rate = state[4], state[5], state[6]
    fraction = ca.fmin(1, ca.fmax(0, (speed - settings.kinematic_blend_speed) /
                                 (settings.dynamic_blend_speed - settings.kinematic_blend_speed)))
    weight = fraction**2 * (3 - 2 * fraction)
    front_force = settings.front_cornering_stiffness * (
        steering - (lateral + settings.front_axle_distance * yaw_rate) / ca.fmax(speed, settings.kinematic_blend_speed))
    loads = (0.5 * settings.air_density * settings.drag_coefficient * settings.frontal_area * speed**2 / settings.mass +
             settings.viscous_resistance * speed / settings.mass + 9.81 * ca.sin(pitch) +
             settings.rolling_resistance_coefficient * 9.81 * ca.cos(pitch) +
             (1 - weight) * yaw_rate * lateral + weight * front_force * ca.sin(steering) / settings.mass)
    request = ca.fmin(settings.maximum_acceleration, ca.fmax(-settings.maximum_deceleration, desired + loads))
    return ca.Function('acceleration_command', [state, desired, steering, pitch], [request])


def dynamic_step(time_step, settings=VehicleConfig(), clip_acceleration=True):
    state = ca.SX.sym('state', 7)
    command = ca.SX.sym('command', 2)
    road = ca.SX.sym('road', 2)
    position_x, position_y, position_z, yaw, speed, lateral_speed, yaw_rate = ca.vertsplit(state)
    acceleration = (ca.fmin(settings.maximum_acceleration,
                            ca.fmax(-settings.maximum_deceleration, command[0]))
                    if clip_acceleration else command[0])
    steering = ca.fmin(settings.maximum_steering,
                      ca.fmax(-settings.maximum_steering, command[1]))
    rear_distance = settings.wheelbase - settings.front_axle_distance
    count = math.ceil(time_step / settings.integration_step)
    substep = time_step / count
    accumulated_longitudinal = 0
    accumulated_lateral = 0

    def blend(forward_speed):
        fraction = ca.fmin(1, ca.fmax(0, (forward_speed - settings.kinematic_blend_speed) /
                                     (settings.dynamic_blend_speed - settings.kinematic_blend_speed)))
        return fraction * fraction * (3 - 2 * fraction)

    for _ in range(count):
        previous_speed, previous_lateral, previous_yaw_rate = speed, lateral_speed, yaw_rate
        weight = blend(previous_speed)
        slip_speed = ca.fmax(previous_speed, settings.kinematic_blend_speed)
        front_force = settings.front_cornering_stiffness * (
            steering - (previous_lateral + settings.front_axle_distance * previous_yaw_rate) / slip_speed)
        aerodynamic = (0.5 * settings.air_density * settings.drag_coefficient * settings.frontal_area *
                       previous_speed**2 / settings.mass)
        viscous = settings.viscous_resistance * previous_speed / settings.mass
        rolling = settings.rolling_resistance_coefficient * 9.81 * ca.cos(road[1])
        free_acceleration = (acceleration - aerodynamic - viscous - 9.81 * ca.sin(road[1]) +
                             weight * (previous_yaw_rate * previous_lateral -
                                       front_force * ca.sin(steering) / settings.mass))
        net = ca.if_else(previous_speed > 0, free_acceleration - rolling,
                         ca.fmax(0, free_acceleration - rolling))
        speed = ca.fmax(0, previous_speed + substep * net)
        next_weight = blend(speed)
        kinematic_yaw_rate = speed * ca.tan(steering) / settings.wheelbase
        kinematic_lateral = rear_distance * kinematic_yaw_rate
        forward_speed = ca.fmax(speed, settings.kinematic_blend_speed)
        front_effective = settings.front_cornering_stiffness * ca.cos(steering)
        coupling = -settings.front_axle_distance * front_effective + rear_distance * settings.rear_cornering_stiffness
        matrix_11 = 1 + substep * (front_effective + settings.rear_cornering_stiffness) / (settings.mass * forward_speed)
        matrix_12 = -substep * (coupling / (settings.mass * forward_speed) - forward_speed)
        matrix_21 = -substep * coupling / (settings.yaw_inertia * forward_speed)
        matrix_22 = 1 + substep * (settings.front_axle_distance**2 * front_effective +
                                 rear_distance**2 * settings.rear_cornering_stiffness) / (settings.yaw_inertia * forward_speed)
        rhs_lateral = previous_lateral + substep * front_effective * steering / settings.mass
        rhs_yaw = previous_yaw_rate + substep * settings.front_axle_distance * front_effective * steering / settings.yaw_inertia
        determinant = matrix_11 * matrix_22 - matrix_12 * matrix_21
        dynamic_lateral = (rhs_lateral * matrix_22 - matrix_12 * rhs_yaw) / determinant
        dynamic_yaw_rate = (matrix_11 * rhs_yaw - matrix_21 * rhs_lateral) / determinant
        lateral_speed = (1 - next_weight) * kinematic_lateral + next_weight * dynamic_lateral
        yaw_rate = (1 - next_weight) * kinematic_yaw_rate + next_weight * dynamic_yaw_rate
        midpoint_yaw_rate = 0.5 * (previous_yaw_rate + yaw_rate)
        midpoint_speed = 0.5 * (previous_speed + speed)
        midpoint_lateral = 0.5 * (previous_lateral + lateral_speed)
        midpoint_yaw = yaw + 0.5 * midpoint_yaw_rate * substep
        position_x += substep * (midpoint_speed * ca.cos(road[0]) * ca.cos(midpoint_yaw) -
                                 midpoint_lateral * ca.sin(midpoint_yaw))
        position_y += substep * (midpoint_speed * ca.cos(road[0]) * ca.sin(midpoint_yaw) +
                                 midpoint_lateral * ca.cos(midpoint_yaw))
        position_z += substep * midpoint_speed * ca.sin(road[0])
        yaw += midpoint_yaw_rate * substep
        accumulated_longitudinal += speed - previous_speed - substep * midpoint_yaw_rate * midpoint_lateral
        accumulated_lateral += lateral_speed - previous_lateral + substep * midpoint_yaw_rate * midpoint_speed
    next_state = ca.vertcat(position_x, position_y, position_z, yaw, speed, lateral_speed, yaw_rate)
    accelerations = ca.vertcat(accumulated_longitudinal / time_step, accumulated_lateral / time_step,
                              (yaw_rate - state[6]) / time_step)
    return ca.Function('dynamic_step', [state, command, road], [next_state, accelerations])