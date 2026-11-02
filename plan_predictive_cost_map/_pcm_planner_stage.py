"""Trajectory-only GP polynomial MPC planner; control belongs to mpc_tracker."""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from collections import deque
from dataclasses import fields
import json
import multiprocessing
from queue import Empty
import signal
from time import monotonic

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from autoware_planning_msgs.msg import Trajectory
from sensor_msgs.msg import PointCloud2
from crp_msgs.msg import Ego, Scenario
from std_msgs.msg import String

from .planner.config import PlannerConfig
from .planner.costmap_visualization import costmap_clouds
from .planner.messages import simulator_parameter_names, simulator_vehicle, snapshot_from_messages, stamp_ns, trajectory_message
from .planner.worker import optimize_worker, put_latest


class PCMPlanner(Node):
    def __init__(self):
        super().__init__('pcm_planner')
        defaults = PlannerConfig()
        self.config = PlannerConfig(**{setting.name: self.declare_parameter(
            setting.name, getattr(defaults, setting.name)).value for setting in fields(defaults)})
        self._egos = deque(maxlen=30)
        self._ego_received = -float('inf')
        self._scenario_received = -float('inf')
        self._snapshot = None
        self._pending_scenario = None
        self._version = 0
        self._submitted = -1
        self._latest_trajectory_stamp = None
        self._debug_packet = None
        self._worker = None
        self._pending_parameters = None
        self._next_parameter_check = 0.
        self._status = {'state': 'waiting_for_simulator'}
        self._process_context = multiprocessing.get_context('spawn')
        self._incoming = self._process_context.Queue(maxsize=1)
        self._outgoing = self._process_context.Queue(maxsize=1)
        self._stop = self._process_context.Event()
        self._trajectory = self.create_publisher(Trajectory, '/plan/trajectory', 1)
        self._diagnostics = self.create_publisher(String, '/plan/pcm_planner/status', 1)
        self._simulator = self.create_client(GetParameters, '/simulator/get_parameters')
        self._sub_ego = self.create_subscription(Ego, '/ego', self._on_ego, 10)
        self._sub_scenario = self.create_subscription(Scenario, '/scenario', self._on_scenario, 1)
        self._timer = self.create_timer(1 / self.config.planning_rate, self._tick)
        self._debug_enabled = self.declare_parameter('visualize_costmap', True).value
        self._debug_rate = self.declare_parameter('costmap_debug_rate', 1.).value
        self._debug_resolution = self.declare_parameter('costmap_resolution', 1.).value
        if self._debug_rate <= 0 or self._debug_resolution <= 0:
            raise ValueError('Cost-map debug rate and resolution must be positive')
        self._clouds = [self.create_publisher(PointCloud2, '/plan/pcm/costmap/t' + str(index), 1)
                        for index in range(4)]
        self._debug_timer = self.create_timer(1 / self._debug_rate, self._publish_costmap)

    def _on_ego(self, message):
        if self._egos and stamp_ns(message.header.stamp) < stamp_ns(self._egos[-1].header.stamp):
            self._egos.clear()
            self._snapshot = None
            self._latest_trajectory_stamp = None
            self._debug_packet = None
        self._egos.append(message)
        self._ego_received = monotonic()
        self._match_inputs()

    def _on_scenario(self, message):
        self._scenario_received = monotonic()
        self._pending_scenario = message
        self._match_inputs()

    def _match_inputs(self):
        if not self._egos or self._pending_scenario is None:
            return
        message = self._pending_scenario
        ego = min(self._egos, key=lambda item: abs(stamp_ns(item.header.stamp) - stamp_ns(message.header.stamp)))
        if abs(stamp_ns(ego.header.stamp) - stamp_ns(message.header.stamp)) > 20_000_000:
            return
        self._pending_scenario = None
        try:
            self._version += 1
            self._snapshot = snapshot_from_messages(message, ego, [0, 0], self._version, self.config)
        except (ValueError, TypeError, IndexError) as error:
            self._snapshot = None
            self._status = {'state': 'invalid_input', 'reason': str(error)}

    def _configure_worker(self, now):
        if self._worker is not None:
            return
        if self._pending_parameters is None:
            if now < self._next_parameter_check or not self._simulator.service_is_ready():
                return
            self._pending_parameters = self._simulator.call_async(GetParameters.Request(names=simulator_parameter_names()))
            return
        if not self._pending_parameters.done():
            return
        try:
            vehicle = simulator_vehicle(self._pending_parameters.result())
            if self.config.maximum_acceleration > vehicle.maximum_acceleration or self.config.maximum_deceleration > vehicle.maximum_deceleration:
                raise ValueError('Planner acceleration bounds exceed simulator limits')
            self._worker = self._process_context.Process(target=optimize_worker,
                args=(self._incoming, self._outgoing, self._stop, self.config, vehicle), daemon=True)
            self._worker.start()
            self._status = {'state': 'compiling_solver', 'model': 'dynamic', 'parameters_from': '/simulator'}
            self.get_logger().info('Building/loading trajectory planner; actuator control belongs to mpc_tracker')
        except Exception as error:
            self._status = {'state': 'blocked', 'reason': str(error)}
        self._pending_parameters = None
        self._next_parameter_check = now + 1.

    def _tick(self):
        now = monotonic()
        self._configure_worker(now)
        while True:
            try:
                update = self._outgoing.get_nowait()
            except Empty:
                break
            if update.get('costmap') is not None:
                self._debug_packet = update
            if 'plan' in update:
                self._latest_trajectory_stamp = update['snapshot'].stamp_ns
                self._status = {'state': 'trajectory_published', 'fit_ms': update['fit_ms'],
                    'total_ms': update['total_ms'], 'solve_ms': update['plan'].solve_seconds * 1000,
                    'solver_status': update['plan'].status, 'iterations': update['plan'].iterations,
                    'objective': update['plan'].objective_components}
                self._trajectory.publish(trajectory_message(update['plan'], update['snapshot'], self.config))
            elif 'error' in update:
                self._status = {'state': 'replan_failed', 'reason': update['error'],
                    'fit_ms': update.get('fit_ms', 0), 'solve_ms': update.get('solve_ms', 0),
                    'validation_ms': update.get('validation_ms', 0), 'total_ms': update.get('total_ms', 0),
                    'deadline_miss': update.get('deadline_miss', False)}
            elif update.get('ready'):
                self._status = {'state': 'ready'}
        fresh = (now - self._ego_received <= self.config.input_timeout and
                 now - self._scenario_received <= self.config.input_timeout and self._snapshot is not None and
                 now - self._snapshot.received_at <= self.config.input_timeout)
        if self._worker is not None and self._worker.is_alive() and fresh and self._snapshot.version != self._submitted:
            put_latest(self._incoming, self._snapshot)
            self._submitted = self._snapshot.version
        status = dict(self._status, input_fresh=fresh, planning_mode=self.config.planning_mode,
                      trajectory_available=self._latest_trajectory_stamp is not None,
                      planning_rate=self.config.planning_rate, horizon=self.config.horizon,
                      planning_deadline_ms=self.config.planning_deadline * 1000)
        self._diagnostics.publish(String(data=json.dumps(status, separators=(',', ':'))))

    def _publish_costmap(self):
        if not self._debug_enabled or self._debug_packet is None:
            return
        packet = self._debug_packet
        times = np.minimum(np.arange(4, dtype=float), self.config.horizon)
        clouds = costmap_clouds(packet['costmap'], packet['snapshot'], self.config,
                                times=times, resolution=self._debug_resolution)
        for publisher, cloud in zip(self._clouds, clouds):
            publisher.publish(cloud)

    def destroy_node(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=1)
            if self._worker.is_alive():
                try:
                    os.killpg(self._worker.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                self._worker.join(timeout=2)
        self._incoming.close()
        self._outgoing.close()
        self._incoming.cancel_join_thread()
        self._outgoing.cancel_join_thread()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PCMPlanner()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()