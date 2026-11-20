"""Vehicle settings for the longitudinal prediction models."""

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