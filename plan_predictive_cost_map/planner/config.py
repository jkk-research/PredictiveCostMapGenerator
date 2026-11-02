"""Explicit vehicle and optimization settings."""

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class VehicleConfig:
    wheelbase: float = 2.7
    mass: float = 1500.0
    yaw_inertia: float = 2500.0
    front_axle_distance: float = 1.2
    front_cornering_stiffness: float = 80000.0
    rear_cornering_stiffness: float = 80000.0
    air_density: float = 1.225
    drag_coefficient: float = 0.3
    frontal_area: float = 2.2
    rolling_resistance_coefficient: float = 0.012
    viscous_resistance: float = 5.0
    kinematic_blend_speed: float = 0.5
    dynamic_blend_speed: float = 2.0
    integration_step: float = 0.005
    maximum_acceleration: float = 9.81
    maximum_deceleration: float = 9.81
    maximum_steering: float = 0.6
    lookahead_time: float = 1.5
    minimum_lookahead: float = 2.0

    def __post_init__(self):
        nonnegative = {'air_density', 'drag_coefficient',
                       'rolling_resistance_coefficient', 'viscous_resistance'}
        for setting in fields(self):
            value = getattr(self, setting.name)
            if not math.isfinite(value) or value < 0 or (value == 0 and setting.name not in nonnegative):
                raise ValueError('Invalid vehicle setting: ' + setting.name)
        if self.front_axle_distance >= self.wheelbase or self.maximum_steering >= math.pi / 2:
            raise ValueError('Invalid axle geometry or steering limit')
        if self.dynamic_blend_speed <= self.kinematic_blend_speed:
            raise ValueError('Dynamic blend speed must exceed kinematic blend speed')


@dataclass(frozen=True)
class PlannerConfig:
    planning_mode: str = 'combined'
    planning_rate: float = 20.0
    horizon: float = 3.0
    control_step: float = 0.05
    cost_step: float = 0.1
    gp_support_points: int = 48
    road_support_points: int = 32
    maximum_branches: int = 24
    maximum_iterations: int = 35
    solve_budget: float = 0.06
    planning_deadline: float = 0.15
    maximum_plan_age: float = 0.30
    input_timeout: float = 0.25
    maximum_acceleration: float = 3.0
    maximum_deceleration: float = 5.0
    maximum_lateral_acceleration: float = 3.0
    maximum_jerk: float = 5.0
    maximum_steering_rate: float = 0.5
    maximum_speed: float = 25.0
    ego_length: float = 4.5
    ego_width: float = 2.0
    safety_margin: float = 0.2
    weight_safety: float = 5.0
    weight_velocity: float = 1.0
    weight_overspeed: float = 1.0
    weight_underspeed: float = 1.0
    overspeed_tolerance: float = 1.0
    weight_acceleration: float = 0.05
    weight_positive_acceleration: float = 1.0
    weight_braking_acceleration: float = 1.0
    weight_lateral_acceleration: float = 0.1
    weight_input_amplitude: float = 0.01
    weight_steering: float = 0.1
    weight_rate: float = 0.1
    weight_crash: float = 100.0
    weight_terminal: float = 5.0
    vehicle_severity: float = 3.0
    pedestrian_severity: float = 3.0
    cyclist_severity: float = 3.0
    obstacle_severity: float = 3.0
    target_cost: float = -1.0
    road_cost: float = -0.4
    gp_amplitude: float = 1.0
    vehicle_length_scale: float = 5.0
    vehicle_width_scale: float = 2.0
    pedestrian_length_scale: float = 1.5
    pedestrian_width_scale: float = 1.5
    time_length_scale: float = 1.0
    route_length_scale: float = 8.0
    route_width_scale: float = 2.0
    road_length_scale: float = 8.0
    road_width_scale: float = 3.0
    risk_limit: float = 0.05
    jit: bool = True

    def __post_init__(self):
        if self.planning_mode not in ('longitudinal', 'combined'):
            raise ValueError('planning_mode must be longitudinal or combined')
        for setting in fields(self):
            value = getattr(self, setting.name)
            if not isinstance(value, (bool, str)) and not math.isfinite(value):
                raise ValueError('Nonfinite planner setting: ' + setting.name)
        for name in ('planning_rate', 'horizon', 'control_step', 'cost_step', 'solve_budget', 'input_timeout',
                 'planning_deadline', 'maximum_plan_age',
                     'overspeed_tolerance',
                     'maximum_acceleration', 'maximum_deceleration', 'maximum_lateral_acceleration',
                     'maximum_jerk', 'maximum_steering_rate', 'maximum_speed', 'ego_length', 'ego_width',
                     'gp_amplitude', 'vehicle_length_scale', 'vehicle_width_scale',
                     'pedestrian_length_scale', 'pedestrian_width_scale', 'time_length_scale',
                     'route_length_scale', 'route_width_scale', 'road_length_scale', 'road_width_scale'):
            if getattr(self, name) <= 0:
                raise ValueError(name + ' must be positive')
        for name in ('gp_support_points', 'road_support_points', 'maximum_branches', 'maximum_iterations'):
            if isinstance(getattr(self, name), bool) or int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 4:
                raise ValueError(name + ' must be an integer of at least four')
        if abs(self.horizon / self.control_step - round(self.horizon / self.control_step)) > 1e-8:
            raise ValueError('Horizon must be a whole number of control steps')
        if self.cost_step < self.control_step or abs(self.cost_step / self.control_step - round(self.cost_step / self.control_step)) > 1e-8:
            raise ValueError('Cost step must be a whole number of control steps')
        if self.planning_deadline >= self.maximum_plan_age:
            raise ValueError('Planning deadline must be less than maximum plan age')
        if self.solve_budget >= self.planning_deadline:
            raise ValueError('Solver budget must leave time for fitting and validation')
        if self.maximum_plan_age >= self.horizon:
            raise ValueError('Maximum plan age must be shorter than the prediction horizon')

    @property
    def steps(self):
        return round(self.horizon / self.control_step)


@dataclass(frozen=True)
class TrackerConfig:
    control_rate: float = 20.0
    horizon: float = 0.75
    time_step: float = 0.05
    maximum_acceleration: float = 3.0
    maximum_deceleration: float = 5.0
    maximum_jerk: float = 5.0
    maximum_steering_rate: float = 0.5
    maximum_speed: float = 40.0
    solve_budget: float = 0.02
    maximum_iterations: int = 1500
    input_timeout: float = 0.25
    maximum_reference_age: float = 2.0
    maximum_position_error: float = 3.0
    maximum_yaw_error: float = 0.5
    maximum_speed_error: float = 3.0
    ego_length: float = 4.5
    ego_width: float = 2.0
    safety_margin: float = 0.3
    virtual_emergency_probability: float = 0.25

    def __post_init__(self):
        for setting in fields(self):
            value = getattr(self, setting.name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid tracker setting: ' + setting.name)
        if abs(self.horizon / self.time_step - round(self.horizon / self.time_step)) > 1e-8:
            raise ValueError('Tracker horizon must contain whole time steps')

    @property
    def steps(self):
        return round(self.horizon / self.time_step)


@dataclass(frozen=True)
class SafetyConfig:
    control_rate: float = 20.0
    input_timeout: float = 0.25
    command_timeout: float = 0.15
    maximum_reference_age: float = 2.0
    maximum_deceleration: float = 5.0
    maximum_jerk: float = 5.0
    ego_length: float = 4.5
    ego_width: float = 2.0
    safety_margin: float = 0.3
    virtual_emergency_probability: float = 0.25

    def __post_init__(self):
        for setting in fields(self):
            value = getattr(self, setting.name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid safety setting: ' + setting.name)
        if self.virtual_emergency_probability > 1:
            raise ValueError('Virtual emergency probability must not exceed one')