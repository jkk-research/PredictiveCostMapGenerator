"""Trajectory/ego tracking MPC; candidate commands are gated by safety_supervisor."""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from dataclasses import fields
import json
from time import monotonic

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from autoware_control_msgs.msg import Control
from autoware_planning_msgs.msg import Trajectory
from crp_msgs.msg import Ego
from std_msgs.msg import String

from .planner.config import TrackerConfig
from .planner.dynamics import acceleration_command
from .planner.messages import measured_state, simulator_parameter_names, simulator_vehicle, stamp_ns, tracking_reference
from .planner.tracking import TrackingFailure, TrackingMPC


class MPCTracker(Node):
    def __init__(self):
        super().__init__('mpc_tracker')
        defaults = TrackerConfig()
        self.config = TrackerConfig(**{setting.name: self.declare_parameter(
            setting.name, getattr(defaults, setting.name)).value for setting in fields(defaults)})
        self._ego = None
        self._ego_received = -float('inf')
        self._reference = None
        self._reference_received = -float('inf')
        self._tracker = None
        self._pending_parameters = None
        self._next_parameter_check = 0.
        self._started = False
        self._stop_acceleration = None
        self._status = {'state': 'waiting_for_simulator'}
        self._commands = self.create_publisher(Control, '/control/tracking/control_cmd', 1)
        self._diagnostics = self.create_publisher(String, '/control/mpc_tracker/status', 1)
        self._simulator = self.create_client(GetParameters, '/simulator/get_parameters')
        self._sub_ego = self.create_subscription(Ego, '/ego', self._on_ego, 10)
        self._sub_reference = self.create_subscription(Trajectory, '/plan/trajectory', self._on_reference, 1)
        self._timer = self.create_timer(1 / self.config.control_rate, self._tick)

    def _on_ego(self, message):
        if self._ego is not None and stamp_ns(message.header.stamp) < stamp_ns(self._ego.header.stamp):
            self._reference = None
            self._started = False
            self._stop_acceleration = None
        self._ego = message
        self._ego_received = monotonic()

    def _on_reference(self, message):
        try:
            reference = tracking_reference(message)
            if self._reference is not None and reference.stamp_ns < self._reference.stamp_ns:
                raise ValueError('Out-of-order reference')
            if self._ego is not None:
                state, _ = measured_state(self._ego)
                elapsed = (stamp_ns(self._ego.header.stamp) - reference.stamp_ns) * 1e-9
                if elapsed < -.1 or elapsed + self.config.horizon > reference.times[-1]:
                    raise ValueError('New reference is already too old or has insufficient horizon')
                target = reference.sample([max(0., elapsed)], state[3])[0][0]
                if np.linalg.norm(target[:2] - state[:2]) > self.config.maximum_position_error:
                    raise ValueError('New reference has a discontinuous position handoff')
                if abs(target[4] - state[4]) > self.config.maximum_speed_error:
                    raise ValueError('New reference has a discontinuous speed handoff')
            self._reference = reference
            self._reference_received = monotonic()
        except (ValueError, TypeError, IndexError) as error:
            self._status = {'state': 'reference_rejected', 'reason': str(error)}

    def _configure(self, now):
        if self._tracker is not None:
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
            self._tracker = TrackingMPC(self.config, self._vehicle)
            self._drive_command = acceleration_command(self._vehicle)
            self._status = {'state': 'ready'}
            self.get_logger().info('Tracking MPC ready; waiting for a validated trajectory')
        except Exception as error:
            self._status = {'state': 'blocked', 'reason': str(error)}
        self._pending_parameters = None
        self._next_parameter_check = now + 1.

    def _tick(self):
        now = monotonic()
        self._configure(now)
        status = dict(self._status, command_valid=False)
        command = None
        if self._ego is not None and self._tracker is not None:
            try:
                state, pitch = measured_state(self._ego)
                if now - self._ego_received > self.config.input_timeout:
                    raise TrackingFailure('Ego input is stale')
                if self._reference is None:
                    raise TrackingFailure('Waiting for a trajectory')
                elapsed = (stamp_ns(self._ego.header.stamp) - self._reference.stamp_ns) * 1e-9
                remaining = self._reference.times[-1] - elapsed
                if now - self._reference_received > self.config.maximum_reference_age:
                    raise TrackingFailure('Reference publication is stale')
                target = self._reference.sample([max(0., elapsed)], state[3])[0][0]
                yaw_error = abs(np.arctan2(np.sin(state[3] - target[3]), np.cos(state[3] - target[3])))
                if np.linalg.norm(state[:2] - target[:2]) > self.config.maximum_position_error or yaw_error > self.config.maximum_yaw_error:
                    raise TrackingFailure('Trajectory tracking error exceeds limits')
                result = self._tracker.solve(state, self._reference, elapsed,
                                              self._ego.accel.accel.linear.x, self._ego.tire_angle_front)
                command = result.command
                self._stop_acceleration = None
                self._started = True
                status = {'state': 'tracking', 'command_valid': True, 'solve_ms': result.solve_seconds * 1000,
                          'total_ms': result.solve_seconds * 1000, 'iterations': result.iterations,
                          'solver_status': result.status, 'reference_age_ms': (now - self._reference_received) * 1000,
                          'remaining_horizon_s': remaining, 'speed_error_mps': state[4] - target[4],
                          'position_error_m': float(np.linalg.norm(state[:2] - target[:2]))}
            except (TrackingFailure, ValueError, RuntimeError) as error:
                state_name = 'controlled_stop' if self._started else ('ready' if self._reference is None else 'waiting')
                status = {'state': state_name, 'reason': str(error), 'command_valid': False}
                if self._started:
                    state, pitch = measured_state(self._ego)
                    if self._stop_acceleration is None:
                        self._stop_acceleration = min(0., self._ego.accel.accel.linear.x)
                    desired = max(-self.config.maximum_deceleration,
                                  self._stop_acceleration - self.config.maximum_jerk / self.config.control_rate)
                    self._stop_acceleration = desired
                    status['requested_net_acceleration'] = desired
                    steering = self._ego.tire_angle_front
                    command = np.array([float(self._drive_command(state, desired, steering, pitch)), steering])
        if command is not None:
            message = Control()
            message.stamp = self.get_clock().now().to_msg()
            message.longitudinal.stamp = message.stamp
            message.lateral.stamp = message.stamp
            message.longitudinal.acceleration = float(command[0])
            message.longitudinal.is_defined_acceleration = True
            message.longitudinal.velocity = max(0., self._ego.twist.twist.linear.x)
            message.lateral.steering_tire_angle = float(command[1])
            self._commands.publish(message)
        status['control_started'] = self._started
        status['control_rate'] = self.config.control_rate
        self._diagnostics.publish(String(data=json.dumps(status, separators=(',', ':'))))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MPCTracker()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()