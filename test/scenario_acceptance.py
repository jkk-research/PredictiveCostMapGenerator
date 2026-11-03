"""Independent ground-truth driving acceptance checks for MAT scenarios."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.io import loadmat
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


def records(value):
    if isinstance(value, dict):
        return [value]
    return [item for item in np.asarray(value, dtype=object).reshape(-1) if isinstance(item, dict)]


def series(value, count, default=0.):
    values = np.asarray(value, dtype=float).reshape(-1)
    if not len(values):
        return np.full(count, default)
    return np.pad(values[:count], (0, max(0, count - len(values))), mode='edge')


def points3(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] not in (2, 3):
        raise ValueError('Invalid scenario centerline')
    return np.column_stack([values, np.zeros(len(values))]) if values.shape[1] == 2 else values


def footprint(position, yaw, dimensions, margin=0.):
    corners = np.array([[-1., -1.], [-1., 1.], [1., 1.], [1., -1.]])
    corners *= [dimensions[0] / 2 + margin, dimensions[1] / 2 + margin]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return Polygon(corners @ rotation.T + np.asarray(position)[:2])


@dataclass
class TruthActor:
    positions: np.ndarray
    headings: np.ndarray
    existence: np.ndarray
    dimensions: np.ndarray
    pedestrian: bool


@dataclass
class ScenarioTruth:
    path: Path
    times: np.ndarray
    route: np.ndarray
    lanes: list
    speeds: np.ndarray
    actors: list
    lane_width: float = 3.85

    @classmethod
    def load(cls, path):
        data = loadmat(path, simplify_cells=True)
        times = np.asarray(data['time'], dtype=float).reshape(-1)
        route = points3(data['targetPath_qg'])
        keep = np.r_[True, np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1) > 1e-8]
        route = route[keep]
        raw_lanes = data['lanes']
        if isinstance(raw_lanes, np.ndarray) and raw_lanes.dtype != object and raw_lanes.ndim == 2:
            raw_lanes = [raw_lanes]
        lanes = [points3(lane) for lane in raw_lanes]
        actors = []
        for name in ('objects', 'obstacles', 'pedestrian', 'cyclists'):
            for actor in records(data.get(name, [])):
                position = actor['position']
                count = min(len(np.asarray(position['x']).reshape(-1)),
                            len(np.asarray(position['y']).reshape(-1)), len(times))
                locations = np.column_stack([series(position.get(axis, []), count) for axis in ('x', 'y', 'z')])
                velocity = actor.get('velocity', [])
                if isinstance(velocity, dict):
                    velocity = velocity.get('velocity', [])
                speeds = series(velocity, count)
                if name != 'obstacles':
                    while count > 1 and np.all(locations[count - 1, :2] == 0) and speeds[count - 1] == 0:
                        count -= 1
                locations = locations[:count]
                delta = np.diff(locations[:, :2], axis=0)
                headings = np.zeros(count)
                if len(delta):
                    moving = np.linalg.norm(delta, axis=1) > 1e-8
                    raw_heading = np.arctan2(delta[:, 1], delta[:, 0])
                    valid = np.flatnonzero(moving)
                    if len(valid):
                        headings[:-1] = np.interp(np.arange(len(delta)), valid, np.unwrap(raw_heading[valid]))
                        headings[-1] = headings[-2]
                size = series(actor.get('size', []), 3, .6 if name == 'pedestrian' else 1.)
                actors.append(TruthActor(locations, headings,
                    series(actor.get('existence_probability', []), count, 1.), size, name == 'pedestrian'))
        speeds = series(data['speedLimit'], len(keep), 0.)[keep]
        return cls(Path(path), times, route, lanes, speeds, actors)

    @property
    def duration(self):
        return float(self.times[-1] - self.times[0])

    @property
    def line(self):
        return LineString(self.route[:, :2])

    @property
    def road_area(self):
        paths = [self.route, *self.lanes]
        polygons = []
        for path in paths:
            if len(path) < 2:
                continue
            first_direction = path[1, :2] - path[0, :2]
            last_direction = path[-1, :2] - path[-2, :2]
            first = path[0, :2] - 3 * first_direction / max(np.linalg.norm(first_direction), 1e-8)
            last = path[-1, :2] + 3 * last_direction / max(np.linalg.norm(last_direction), 1e-8)
            polygons.append(LineString(np.vstack([first, path[:, :2], last])).buffer(
                self.lane_width / 2, cap_style='flat', join_style='mitre'))
        return unary_union(polygons)

    def actors_at(self, elapsed):
        index = min(int(np.floor(elapsed / (self.times[1] - self.times[0]) + 1e-6)), len(self.times) - 1)
        return [(actor, actor.positions[index], actor.headings[index]) for actor in self.actors
                if index < len(actor.positions) and actor.existence[index] > 0]


@dataclass(frozen=True)
class Limits:
    ego_length: float = 4.5
    ego_width: float = 2.
    ego_height: float = 1.5
    road_tolerance: float = .15
    maximum_acceleration: float = 3.5
    maximum_deceleration: float = 5.5
    maximum_lateral_acceleration: float = 3.5
    maximum_jerk: float = 6.
    speed_tolerance: float = .5
    progress_fraction: float = .90
    return_to_path_tolerance: float = .75
    maximum_unjustified_excursion_seconds: float = 3.
    maximum_clear_road_stall_seconds: float = 15.


@dataclass
class DrivingSample:
    elapsed: float
    position: np.ndarray
    yaw: float
    speed: float
    acceleration: float
    lateral_acceleration: float
    yaw_rate: float
    uncertain_hazards: bool = False


class AcceptanceMonitor:
    def __init__(self, truth, limits=Limits()):
        self.truth, self.limits = truth, limits
        self.route = truth.line
        self.area = truth.road_area.buffer(limits.road_tolerance)
        self.arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(truth.route[:, :2], axis=0), axis=1))]
        headings = np.unwrap(np.arctan2(np.gradient(truth.route[:, 1]), np.gradient(truth.route[:, 0])))
        self.curvature = np.gradient(headings, self.arc)
        self.previous = None
        self.failures = []
        self.samples = 0
        self.progress = 0.
        self.maximum_cross_track = 0.
        self.maximum_jerk = 0.
        self.maximum_lateral = 0.
        self.minimum_clearance = float('inf')
        self.pedestrian_checks = 0
        self.pedestrian_stops = 0
        self.excursion = False
        self.completed = False
        self.unjustified_excursion_since = None
        self.clear_progress_since = None
        self.clear_progress_station = 0.
        self.was_moving = False
        self.unjustified_stop = False
        self.stop_restart_cycles = 0

    def fail(self, criterion, sample, detail):
        if not any(item['criterion'] == criterion for item in self.failures):
            self.failures.append({'criterion': criterion, 'time_s': round(sample.elapsed, 3), 'detail': detail})

    def observe(self, sample, check_comfort=True):
        self.samples += 1
        values = np.r_[sample.position, sample.yaw, sample.speed, sample.acceleration,
                       sample.lateral_acceleration, sample.yaw_rate]
        if not np.isfinite(values).all():
            self.fail('finite_state', sample, 'Nonfinite ego state')
            return
        ego_shape = footprint(sample.position, sample.yaw, [self.limits.ego_length, self.limits.ego_width])
        station = self.route.project(Point(sample.position[:2]))
        cross_track = self.route.distance(Point(sample.position[:2]))
        self.progress = max(self.progress, station)
        self.maximum_cross_track = max(self.maximum_cross_track, cross_track)
        if cross_track > (self.truth.lane_width - self.limits.ego_width) / 2 + self.limits.road_tolerance:
            self.excursion = True
        if not self.area.covers(ego_shape):
            self.fail('drivable_surface', sample, 'Ego footprint leaves the supplied drivable surface')
        limit = float(np.interp(station, self.arc, self.truth.speeds))
        if sample.speed > limit + self.limits.speed_tolerance:
            self.fail('speed_limit', sample, 'Speed %.2f exceeds %.2f m/s' % (sample.speed, limit))
        lateral = max(abs(sample.lateral_acceleration), abs(sample.speed * sample.yaw_rate))
        self.maximum_lateral = max(self.maximum_lateral, lateral)
        if lateral > self.limits.maximum_lateral_acceleration:
            self.fail('curve_speed', sample, 'Lateral acceleration %.2f m/s^2' % lateral)
        if cross_track < .75:
            curve_load = sample.speed**2 * abs(float(np.interp(station, self.arc, self.curvature)))
            if curve_load > self.limits.maximum_lateral_acceleration + .5:
                self.fail('curve_speed', sample, 'Speed excessive for target-path curvature')
        if check_comfort:
            if sample.acceleration > self.limits.maximum_acceleration or sample.acceleration < -self.limits.maximum_deceleration:
                self.fail('longitudinal_comfort', sample, 'Longitudinal acceleration %.2f m/s^2' % sample.acceleration)
            if self.previous is not None and sample.elapsed > self.previous.elapsed:
                jerk = abs(sample.acceleration - self.previous.acceleration) / (sample.elapsed - self.previous.elapsed)
                self.maximum_jerk = max(self.maximum_jerk, jerk)
                if jerk > self.limits.maximum_jerk:
                    self.fail('smoothness', sample, 'Longitudinal jerk %.2f m/s^3' % jerk)
        nearby_hazard = sample.uncertain_hazards
        for actor, location, heading in self.truth.actors_at(sample.elapsed):
            if sample.position[2] >= location[2] + actor.dimensions[2] or location[2] >= sample.position[2] + self.limits.ego_height:
                continue
            actor_shape = footprint(location, heading, actor.dimensions)
            self.minimum_clearance = min(self.minimum_clearance, ego_shape.distance(actor_shape))
            if ego_shape.intersects(actor_shape):
                self.fail('collision', sample, 'Ground-truth %s footprint overlaps ego' % ('pedestrian' if actor.pedestrian else 'actor'))
            actor_station = self.route.project(Point(location[:2]))
            on_target = self.route.distance(Point(location[:2])) < self.truth.lane_width / 2 + actor.dimensions[1] / 2
            ahead = actor_station - station
            if -10. < ahead < max(50., sample.speed * 4) and self.route.distance(Point(location[:2])) < 2 * self.truth.lane_width:
                nearby_hazard = True
            if actor.pedestrian and on_target and -2. < ahead < max(15., sample.speed**2 / 10 + 2 * sample.speed):
                self.pedestrian_checks += 1
                if sample.speed < .15 and ahead > self.limits.ego_length / 2:
                    self.pedestrian_stops += 1
                if ahead > 0 and ahead < self.limits.ego_length / 2 + actor.dimensions[0] / 2 + .5 and sample.speed > .15:
                    self.fail('pedestrian_yield', sample, 'Ego did not stop before an occupied target-path crossing')
        off_target = cross_track > (self.truth.lane_width - self.limits.ego_width) / 2 + self.limits.road_tolerance
        if off_target and not nearby_hazard:
            if self.unjustified_excursion_since is None:
                self.unjustified_excursion_since = sample.elapsed
            if sample.elapsed - self.unjustified_excursion_since > self.limits.maximum_unjustified_excursion_seconds:
                self.fail('lane_discipline', sample, 'Sustained departure from target lane without a nearby hazard')
        else:
            self.unjustified_excursion_since = None
        if nearby_hazard or self.progress - self.clear_progress_station >= 1.:
            self.clear_progress_since = sample.elapsed
            self.clear_progress_station = self.progress
        elif self.clear_progress_since is None:
            self.clear_progress_since = sample.elapsed
        elif sample.elapsed - self.clear_progress_since > self.limits.maximum_clear_road_stall_seconds:
            self.fail('progress', sample, 'No meaningful progress on a clear road for 15 seconds')
        if sample.speed > 1.:
            if self.unjustified_stop:
                self.stop_restart_cycles += 1
                self.fail('stop_restart', sample, 'Stopped and restarted without a nearby real or virtual hazard')
            self.was_moving = True
            self.unjustified_stop = False
        elif sample.speed < .15 and self.was_moving:
            self.unjustified_stop = not nearby_hazard
            self.was_moving = False
        if station >= self.limits.progress_fraction * self.route.length:
            self.completed = True
            if cross_track > self.limits.return_to_path_tolerance:
                self.fail('return_to_target', sample, 'Passing excursion did not return toward the target route')
        self.previous = sample

    def report(self, elapsed, termination):
        failures = list(self.failures)
        if not self.completed and termination != 'first_acceptance_failure':
            failures.append({'criterion': 'progress', 'time_s': round(elapsed, 3),
                             'detail': 'Only %.1f%% of route reached (%s)' % (100 * self.progress / self.route.length, termination)})
        return {'scenario': self.truth.path.name, 'status': 'FAIL' if failures else 'PASS',
                'termination': termination, 'failures': failures,
                'metrics': {'simulated_seconds': elapsed, 'samples': self.samples,
                    'progress_fraction': self.progress / self.route.length,
                    'maximum_cross_track_m': self.maximum_cross_track,
                    'maximum_jerk_mps3': self.maximum_jerk, 'maximum_lateral_acceleration_mps2': self.maximum_lateral,
                    'minimum_clearance_m': None if np.isinf(self.minimum_clearance) else self.minimum_clearance,
                    'intentional_excursion_observed': self.excursion, 'pedestrian_checks': self.pedestrian_checks,
                    'unjustified_stop_restart_cycles': self.stop_restart_cycles,
                    'pedestrian_stopped_samples': self.pedestrian_stops},
                'coverage': {'full_route': self.completed, 'pedestrian_conflict_encountered': self.pedestrian_checks > 0,
                             'progress_status': 'PASS' if self.completed else ('NOT_EVALUATED' if termination == 'first_acceptance_failure' else 'FAIL'),
                             'overtaking_required': 'Scenario-dependent; safe stopping alone does not satisfy progress'}}


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
    try:
        return process.communicate(timeout=10)[0]
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return process.communicate()[0]


def run_case(path, root, initial_speed=50 / 3.6, fail_fast=True, maximum_seconds=None, planning_mode='combined'):
    import rclpy
    from crp_msgs.msg import Ego
    from crp_msgs.msg import Scenario
    from autoware_control_msgs.msg import Control
    from std_msgs.msg import String
    truth = ScenarioTruth.load(path)
    monitor = AcceptanceMonitor(truth)
    time_limit = maximum_seconds or (truth.duration + truth.line.length / max(np.mean(truth.speeds), 1.) + 20.)
    os.environ['ROS_DOMAIN_ID'] = str(120 + os.getpid() % 80)
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    rclpy.init()
    node = rclpy.create_node('scenario_acceptance')
    statuses = []
    planner_statuses = []
    safety_statuses = []
    active_at = None
    ordinal = 0
    first_speed = None
    uncertain_hazards = False
    control_started = False
    elapsed = 0.

    def on_control(message):
        nonlocal control_started
        control_started = True

    def on_scenario(message):
        nonlocal uncertain_hazards
        uncertain_hazards = bool(message.virtual_object_ids)

    def on_ego(message):
        nonlocal ordinal, active_at, first_speed, elapsed
        elapsed = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        ordinal += 1
        first_speed = message.twist.twist.linear.x if first_speed is None else first_speed
        orientation = message.pose.pose.orientation
        yaw = np.arctan2(2 * (orientation.w * orientation.z + orientation.x * orientation.y),
                         1 - 2 * (orientation.y**2 + orientation.z**2))
        sample = DrivingSample(elapsed, np.array([message.pose.pose.position.x, message.pose.pose.position.y,
                                                 message.pose.pose.position.z]), yaw, message.twist.twist.linear.x,
                               message.accel.accel.linear.x, message.accel.accel.linear.y,
                               message.twist.twist.angular.z, uncertain_hazards)
        if statuses and statuses[-1].get('command_valid') and active_at is None:
            active_at = elapsed
        monitor.observe(sample, active_at is not None and elapsed > active_at + .5)

    subscriptions = [node.create_subscription(Ego, '/ego', on_ego, 1000),
                     node.create_subscription(Control, '/control/command/control_cmd', on_control, 1000),
                     node.create_subscription(Scenario, '/scenario', on_scenario, 10),
                     node.create_subscription(String, '/control/mpc_tracker/status', lambda value: statuses.append(json.loads(value.data)), 1000),
                     node.create_subscription(String, '/plan/pcm_planner/status', lambda value: planner_statuses.append(json.loads(value.data)), 1000),
                     node.create_subscription(String, '/control/safety_supervisor/status', lambda value: safety_statuses.append(json.loads(value.data)), 1000)]
    scenario_generator = subprocess.Popen([
        str(root / 'install/scenario_generation/lib/scenario_generation/scenario_generator'),
        '--ros-args', '-p', 'scenario_file:=' + str(path),
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    simulator = subprocess.Popen([str(root / 'install/simulator/lib/simulator/simulator'), '--ros-args',
        '-p', 'scenario_file:=' + str(path), '-p', 'command_mode:=acceleration', '-p', 'vehicle_model:=dynamic',
        '-p', 'use_internal_controller:=false', '-p', 'simulation_time_stamps:=true',
        '-p', 'use_scenario_speed_when_uncommanded:=false',
        '-p', 'initial_speed:=' + str(float(initial_speed)), '-p', 'stop_at_scenario_end:=false'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    planner = subprocess.Popen([str(root / 'install/plan_predictive_cost_map/lib/plan_predictive_cost_map/pcm_planner'),
        '--ros-args', '--params-file', str(root / 'install/plan_predictive_cost_map/share/plan_predictive_cost_map/config/pcm_planner.yaml'),
        '-p', 'planning_mode:=' + planning_mode],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    tracker = subprocess.Popen([str(root / 'install/plan_predictive_cost_map/lib/plan_predictive_cost_map/mpc_tracker')],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    safety = subprocess.Popen([str(root / 'install/plan_predictive_cost_map/lib/plan_predictive_cost_map/safety_supervisor')],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    termination = 'timeout'
    outputs = []
    try:
        wall_limit = time.monotonic() + time_limit + 15
        while time.monotonic() < wall_limit:
            if any(process.poll() is not None for process in (scenario_generator, simulator, planner, tracker, safety)):
                termination = 'process_exit'
                break
            rclpy.spin_once(node, timeout_sec=.02)
            if monitor.failures and fail_fast:
                termination = 'first_acceptance_failure'
                break
            if monitor.completed:
                termination = 'route_complete'
                break
            if elapsed >= time_limit:
                break
        if first_speed is None or abs(first_speed - initial_speed) > 1e-4:
            monitor.failures.append({'criterion': 'initial_speed', 'time_s': 0., 'detail': 'Initial speed did not match the configured value'})
        report = monitor.report(elapsed, termination)
        report['initial_speed_mps'] = initial_speed
        report['planning_mode'] = planning_mode
        report['architecture'] = 'trajectory_planner_mpc_tracker_safety_supervisor'
        report['fallback_reasons'] = dict(Counter('planning_deadline' if value.get('deadline_miss') else
            value.get('reason', value.get('state')).split(';')[0] for value in statuses if not value.get('command_valid')))
        report['limits'] = vars(monitor.limits)
        report['safety_interventions'] = dict(Counter(value.get('reason', value['state']) for value in safety_statuses if value.get('intervening')))
        report['last_planner_status'] = planner_statuses[-1] if planner_statuses else None
        report['final_command_publishers'] = sorted({entry.node_name for entry in node.get_publishers_info_by_topic('/control/command/control_cmd')})
        return report
    finally:
        outputs.append(stop_process(safety))
        outputs.append(stop_process(tracker))
        outputs.append(stop_process(planner))
        outputs.append(stop_process(simulator))
        outputs.append(stop_process(scenario_generator))
        node.destroy_node()
        rclpy.shutdown()
        if any('Traceback' in output for output in outputs):
            print('\n'.join(output[-2000:] for output in outputs), file=sys.stderr)


def warm_solver(planning_mode='combined'):
    with tempfile.TemporaryDirectory(prefix='pcm_acceptance_warmup_') as directory:
        subprocess.run([sys.executable, '-c',
            'from plan_predictive_cost_map.planner.optimizer import PolynomialMPC; '
            'from plan_predictive_cost_map.planner.config import PlannerConfig; '
            'from dataclasses import fields; from pathlib import Path; import yaml; '
            'settings=yaml.safe_load(Path(' + repr(str(Path(__file__).resolve().parents[3] / 'install/plan_predictive_cost_map/share/plan_predictive_cost_map/config/pcm_planner.yaml')) + ').read_text())["pcm_planner"]["ros__parameters"]; '
            'settings={key:value for key,value in settings.items() if key in {item.name for item in fields(PlannerConfig)}}; '
            'settings["planning_mode"]=' + repr(planning_mode) + '; PolynomialMPC(PlannerConfig(**settings))'],
            cwd=directory, check=True, timeout=360)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', action='append', default=[])
    parser.add_argument('--report', type=Path, default=Path('/tmp/pcm_scenario_acceptance.json'))
    parser.add_argument('--initial-speed', type=float, default=50 / 3.6)
    parser.add_argument('--planning-mode', choices=['longitudinal', 'combined'], default='combined')
    parser.add_argument('--maximum-seconds', type=float)
    parser.add_argument('--continue-after-failure', action='store_true')
    parser.add_argument('--skip-warmup', action='store_true')
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    cases = sorted((root / 'src/scenario_files').glob('*.mat'))
    if arguments.scenario:
        cases = [path for path in cases if any(name in path.name for name in arguments.scenario)]
    if not cases:
        raise SystemExit('No matching scenarios')
    if not arguments.skip_warmup:
        warm_solver(arguments.planning_mode)
    reports = []
    for case in cases:
        report = run_case(case, root, arguments.initial_speed, not arguments.continue_after_failure,
                          arguments.maximum_seconds, arguments.planning_mode)
        reports.append(report)
        print(case.name, report['status'], report['failures'], flush=True)
        arguments.report.write_text(json.dumps(reports, indent=2, allow_nan=False) + '\n')
    return int(any(report['status'] != 'PASS' for report in reports))


if __name__ == '__main__':
    raise SystemExit(main())