"""Independent safety guard and sole final actuator-command publisher."""

from collections import deque
from copy import deepcopy
from dataclasses import fields
import json
from time import monotonic

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from autoware_control_msgs.msg import Control
from autoware_planning_msgs.msg import Trajectory
from crp_msgs.msg import Ego, Scenario
from std_msgs.msg import String

from .planner.config import PlannerConfig, SafetyConfig
from .planner.dynamics import acceleration_command
from .planner.geometry import trajectory_overlaps
from .planner.messages import measured_state, simulator_parameter_names, simulator_vehicle, snapshot_from_messages, stamp_ns, tracking_reference


def intersecting_hazard(snapshot, reference, model_time_ns, config):
    elapsed = (model_time_ns - reference.stamp_ns) * 1e-9
    if elapsed < -.1 or elapsed >= reference.times[-1]:
        raise ValueError('Active trajectory has no usable future horizon')
    remaining = reference.times[-1] - max(0., elapsed)
    horizon = min(remaining, max(1., snapshot.state[4] / config.maximum_deceleration + .5))
    times = np.arange(0., max(0., horizon) + .025, .05)
    predicted = reference.sample(max(0., elapsed) + times, snapshot.state[3])[0]
    snapshot_age = max(0., (model_time_ns - snapshot.stamp_ns) * 1e-9)
    for branch in snapshot.branches:
        if branch.virtual and branch.existence * branch.confidence < config.virtual_emergency_probability:
            continue
        if trajectory_overlaps(predicted, branch.sample(snapshot_age + times), branch.headings(snapshot_age + times),
                               [config.ego_length, config.ego_width], branch.dimensions, config.safety_margin):
            return 'virtual' if branch.virtual else 'real'
    return None


class SafetySupervisor(Node):
    def __init__(self):
        super().__init__('safety_supervisor')
        defaults = SafetyConfig()
        self.config = SafetyConfig(**{setting.name: self.declare_parameter(
            setting.name, getattr(defaults, setting.name)).value for setting in fields(defaults)})
        self._egos = deque(maxlen=30)
        self._ego = None
        self._ego_received = -float('inf')
        self._scenario_received = -float('inf')
        self._command_received = -float('inf')
        self._reference_received = -float('inf')
        self._candidate = None
        self._reference = None
        self._pending_scenario = None
        self._snapshot = None
        self._drive_command = None
        self._pending_parameters = None
        self._next_parameter_check = 0.
        self._started = False
        self._stop_acceleration = None
        self._adapter_config = PlannerConfig(maximum_branches=128, gp_support_points=256)
        self._configuration_error = None
        self._commands = self.create_publisher(Control, '/control/command/control_cmd', 1)
        self._diagnostics = self.create_publisher(String, '/control/safety_supervisor/status', 1)
        self._simulator = self.create_client(GetParameters, '/simulator/get_parameters')
        self._sub_ego = self.create_subscription(Ego, '/ego', self._on_ego, 10)
        self._sub_scenario = self.create_subscription(Scenario, '/simulator/tracking_scenario', self._on_scenario, 1)
        self._sub_reference = self.create_subscription(Trajectory, '/plan/trajectory', self._on_reference, 1)
        self._sub_candidate = self.create_subscription(Control, '/control/tracking/control_cmd', self._on_candidate, 1)
        self._timer = self.create_timer(1 / self.config.control_rate, self._tick)

    def _on_ego(self, message):
        if self._ego is not None and stamp_ns(message.header.stamp) < stamp_ns(self._ego.header.stamp):
            self._egos.clear()
            self._snapshot = None
            self._candidate = None
            self._reference = None
            self._started = False
            self._stop_acceleration = None
        self._ego = message
        self._egos.append(message)
        self._ego_received = monotonic()
        self._match_scenario()

    def _on_scenario(self, message):
        self._pending_scenario = message
        self._scenario_received = monotonic()
        self._match_scenario()

    def _match_scenario(self):
        if self._pending_scenario is None or not self._egos:
            return
        message = self._pending_scenario
        ego = min(self._egos, key=lambda item: abs(stamp_ns(item.header.stamp) - stamp_ns(message.header.stamp)))
        if abs(stamp_ns(ego.header.stamp) - stamp_ns(message.header.stamp)) > 20_000_000:
            return
        self._pending_scenario = None
        try:
            self._snapshot = snapshot_from_messages(message, ego, [0, 0], 0, self._adapter_config, safety_only=True)
        except (ValueError, TypeError, IndexError):
            self._snapshot = None

    def _on_reference(self, message):
        try:
            reference = tracking_reference(message)
            if self._reference is not None and reference.stamp_ns < self._reference.stamp_ns:
                return
            self._reference = reference
            self._reference_received = monotonic()
        except (ValueError, TypeError, IndexError):
            pass

    def _on_candidate(self, message):
        if not message.longitudinal.is_defined_acceleration or not np.isfinite([
            message.longitudinal.acceleration, message.lateral.steering_tire_angle,
            message.longitudinal.velocity]).all():
            self._candidate = None
            return
        self._candidate = message
        self._command_received = monotonic()

    def _configure(self, now):
        if self._drive_command is not None:
            return
        if self._pending_parameters is None:
            if now < self._next_parameter_check or not self._simulator.service_is_ready():
                return
            self._pending_parameters = self._simulator.call_async(GetParameters.Request(names=simulator_parameter_names()))
            return
        if not self._pending_parameters.done():
            return
        try:
            self._vehicle = simulator_vehicle(self._pending_parameters.result())
            self._drive_command = acceleration_command(self._vehicle)
            self._configuration_error = None
        except Exception as error:
            self._configuration_error = str(error)
        self._pending_parameters = None
        self._next_parameter_check = now + 1.

    def _select_command(self, now):
        reason = self._configuration_error
        hazard = None
        if self._ego is None or self._drive_command is None:
            return None, {'state': 'waiting_for_simulator', 'reason': reason, 'intervening': False}
        state, pitch = measured_state(self._ego)
        if now - self._ego_received > self.config.input_timeout:
            reason = 'Ego input is stale'
        elif self._snapshot is None or now - self._scenario_received > self.config.input_timeout or now - self._snapshot.received_at > self.config.input_timeout:
            reason = 'Safety scenario is stale or invalid'
        elif self._candidate is None or now - self._command_received > self.config.command_timeout:
            reason = 'Tracking command is stale or invalid'
        elif self._reference is None or now - self._reference_received > self.config.maximum_reference_age:
            reason = 'Reference is missing or stale'
        else:
            try:
                hazard = intersecting_hazard(self._snapshot, self._reference, stamp_ns(self._ego.header.stamp), self.config)
                if hazard is not None:
                    reason = 'Updated ' + hazard + ' hazard intersects the active trajectory'
            except ValueError as error:
                reason = str(error)
        if reason is None:
            command = deepcopy(self._candidate)
            if not -self._vehicle.maximum_deceleration - 1e-6 <= command.longitudinal.acceleration <= self._vehicle.maximum_acceleration + 1e-6 or abs(command.lateral.steering_tire_angle) > self._vehicle.maximum_steering + 1e-6:
                reason = 'Tracking command exceeds actuator bounds'
            else:
                self._started = True
                self._stop_acceleration = None
                return command, {'state': 'pass_through', 'intervening': False}
        if not self._started and self._candidate is None:
            return None, {'state': 'waiting', 'reason': reason, 'intervening': False}
        self._started = True
        if self._stop_acceleration is None:
            self._stop_acceleration = min(0., self._ego.accel.accel.linear.x)
        desired = -self.config.maximum_deceleration if hazard is not None else max(
            -self.config.maximum_deceleration,
            self._stop_acceleration - self.config.maximum_jerk / self.config.control_rate)
        self._stop_acceleration = desired
        command = Control()
        command.longitudinal.acceleration = float(self._drive_command(state, desired, self._ego.tire_angle_front, pitch))
        command.longitudinal.is_defined_acceleration = True
        command.longitudinal.velocity = 0.
        command.lateral.steering_tire_angle = float(np.clip(self._ego.tire_angle_front,
                                                          -self._vehicle.maximum_steering, self._vehicle.maximum_steering))
        return command, {'state': 'emergency_brake' if hazard is not None else 'controlled_stop',
                         'reason': reason, 'intervening': True, 'requested_net_acceleration': desired}

    def _tick(self):
        now = monotonic()
        self._configure(now)
        try:
            command, status = self._select_command(now)
        except (ValueError, RuntimeError) as error:
            command = None
            status = {'state': 'invalid_input', 'reason': str(error), 'intervening': True}
            if self._started and self._drive_command is not None:
                command = Control()
                command.longitudinal.acceleration = -self._vehicle.maximum_deceleration
                command.longitudinal.is_defined_acceleration = True
        if command is not None:
            command.stamp = self.get_clock().now().to_msg()
            command.longitudinal.stamp = command.stamp
            command.lateral.stamp = command.stamp
            self._commands.publish(command)
        status['control_started'] = self._started
        status['command_forwarded'] = command is not None
        self._diagnostics.publish(String(data=json.dumps(status, separators=(',', ':'))))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SafetySupervisor()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()