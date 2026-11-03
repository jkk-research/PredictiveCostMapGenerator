"""ROS message conversion and timestamp-anchored map-frame snapshots."""

from dataclasses import dataclass, fields
from time import monotonic

import numpy as np
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .costmap import Branch
from .geometry import Road, resample_polyline
from .config import VehicleConfig


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def quaternion(pose):
    value = pose.orientation
    return Rotation.from_quat([value.x, value.y, value.z, value.w])


def position(point):
    return [point.x, point.y, point.z]


def simulator_parameter_names():
    aliases = {'mass': 'vehicle_mass', 'integration_step': 'dynamics_integration_step'}
    return ([aliases.get(setting.name, setting.name) for setting in fields(VehicleConfig)] +
            ['command_mode', 'vehicle_model', 'use_internal_controller', 'use_road_gradient',
             'road_pitch', 'simulation_time_stamps', 'simulation_step'])


def simulator_vehicle(response):
    names = simulator_parameter_names()
    if len(response.values) != len(names):
        raise ValueError('Simulator parameter response is incomplete')
    values = []
    for value in response.values:
        attributes = {1: 'bool_value', 2: 'integer_value', 3: 'double_value', 4: 'string_value'}
        if value.type not in attributes:
            raise ValueError('Simulator is missing required planner/tracker parameters')
        values.append(getattr(value, attributes[value.type]))
    parameters = dict(zip(names, values))
    if parameters['command_mode'] != 'acceleration' or parameters['vehicle_model'] != 'dynamic' or parameters['use_internal_controller']:
        raise ValueError('Simulator requires dynamic model, acceleration mode, and external steering')
    if not parameters['use_road_gradient'] or abs(parameters['road_pitch']) > 1e-9:
        raise ValueError('Planner/tracker requires automatic gradient and zero constant road pitch')
    if not parameters['simulation_time_stamps']:
        raise ValueError('Enable simulation_time_stamps for consistent trajectory timing')
    if abs(parameters['simulation_step'] - .05) > 1e-9:
        raise ValueError('Planner/tracker currently requires simulation_step=0.05')
    aliases = {'mass': 'vehicle_mass', 'integration_step': 'dynamics_integration_step'}
    return VehicleConfig(**{setting.name: parameters[aliases.get(setting.name, setting.name)]
                            for setting in fields(VehicleConfig)})


def measured_state(ego):
    rotation = quaternion(ego.pose.pose)
    angles = rotation.as_euler('xyz')
    state = np.r_[position(ego.pose.pose.position), angles[2], ego.twist.twist.linear.x,
                  ego.twist.twist.linear.y, ego.twist.twist.angular.z]
    if not np.isfinite(state).all() or state[4] < 0:
        raise ValueError('Invalid measured ego state')
    return state, -angles[1]


def tracking_reference(message):
    from .tracking import TrackingReference
    if message.header.frame_id not in ('map', 'map_local') or len(message.points) < 2:
        raise ValueError('Tracker requires a nonempty map or map_local trajectory')
    states, pitches, times, accelerations, steering = [], [], [], [], []
    for point in message.points:
        angles = quaternion(point.pose).as_euler('xyz')
        states.append([*position(point.pose.position), angles[2], point.longitudinal_velocity_mps,
                       point.lateral_velocity_mps, point.heading_rate_rps])
        pitches.append(-angles[1])
        times.append(point.time_from_start.sec + point.time_from_start.nanosec * 1e-9)
        accelerations.append(point.acceleration_mps2)
        steering.append(point.front_wheel_angle_rad)
    states = np.asarray(states)
    states[:, 3] = np.unwrap(states[:, 3])
    return TrackingReference(stamp_ns(message.header.stamp), np.asarray(times), states,
                              np.asarray(pitches), np.asarray(accelerations), np.asarray(steering))


@dataclass(frozen=True)
class Snapshot:
    version: int
    stamp_ns: int
    received_at: float
    state: np.ndarray
    previous_command: np.ndarray
    road: Road
    lanes: tuple
    branches: tuple
    frame_id: str = 'map'


def snapshot_from_messages(scenario, ego, previous_command, version, config, safety_only=False):
    ego_frame = ego.header.frame_id
    scenario_frame = scenario.header.frame_id
    if ego_frame not in ('map', 'map_local') or scenario_frame not in (ego_frame, 'base_link'):
        raise ValueError('Planner requires map or map_local ego and a matching or base_link scenario')
    if abs(stamp_ns(scenario.header.stamp) - stamp_ns(ego.header.stamp)) > 20_000_000:
        raise ValueError('Scenario and ego timestamps differ by more than 20 ms')
    rotation = quaternion(ego.pose.pose)
    translation = np.array(position(ego.pose.pose.position))
    yaw = rotation.as_euler('xyz')[2]

    def convert(points):
        values = np.asarray(points, dtype=float).reshape(-1, 3)
        if scenario.header.frame_id == 'base_link':
            values = rotation.apply(values) + translation
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite scenario geometry')
        return values

    if not scenario.paths or len(scenario.paths[0].path.points) < 2:
        raise ValueError('Target route is missing')
    target = scenario.paths[0]
    target_points = [target.path.points[0], target.path.points[-1]] if safety_only else target.path.points
    route = convert([position(point.point.pose.position) for point in target_points])
    widths = [np.linalg.norm(np.subtract(position(left), position(right)))
              for left, right in zip(target.path.left_bound[:1] if safety_only else target.path.left_bound,
                                     target.path.right_bound[:1] if safety_only else target.path.right_bound)]
    if not widths:
        raise ValueError('Target corridor boundaries are missing')
    width = float(np.median(widths))
    speed = float(target.traffic_rules.maximum_speed)
    if not np.isfinite([width, speed]).all() or width <= 0 or speed < 0:
        raise ValueError('Invalid corridor width or speed limit')
    areas = []
    for corridor in (() if safety_only else scenario.paths):
        path = corridor.path
        if len(path.left_bound) >= 2 and len(path.right_bound) >= 2:
            left = convert([position(point) for point in path.left_bound])
            right = convert([position(point) for point in path.right_bound])
            tangent = left[1] + right[1] - left[0] - right[0]
            padding = (config.ego_length / 2 + config.safety_margin + 1e-3) * tangent / max(np.linalg.norm(tangent[:2]), 1e-12)
            left = np.vstack([left[0] - padding, left])
            right = np.vstack([right[0] - padding, right])
            polygon = Polygon(np.vstack([left[:, :2], right[::-1, :2]]))
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty:
                areas.append(polygon)
    if not areas and not safety_only:
        raise ValueError('Drivable road polygons are missing')
    road = Road(route if safety_only else resample_polyline(route, config.road_support_points),
                width, speed, None if safety_only else unary_union(areas))
    lanes = []
    for corridor in (() if safety_only else scenario.paths[1:]):
        if len(corridor.path.points) >= 2:
            lanes.append(resample_polyline(convert([position(point.point.pose.position)
                                                    for point in corridor.path.points]), config.road_support_points))
    virtual_ids = {bytes(identifier.uuid) for identifier in scenario.virtual_object_ids}
    branches = []
    for objects, obstacle in ((scenario.local_moving_objects.objects, False),
                              (scenario.local_obstacles.objects, True)):
        for actor in objects:
            existence = float(actor.existence_probability)
            if not np.isfinite(existence) or not 0 <= existence <= 1:
                raise ValueError('Invalid existence probability')
            if existence == 0:
                continue
            label = max(actor.classification, key=lambda entry: entry.probability).label if actor.classification else 0
            kind = 3 if obstacle else (1 if label == 7 else (2 if label == 6 else 0))
            source_pose = actor.kinematics.initial_pose_with_covariance.pose
            source_rotation = quaternion(source_pose)
            map_rotation = rotation * source_rotation if scenario.header.frame_id == 'base_link' else source_rotation
            heading = map_rotation.as_euler('xyz')[2]
            dimensions = np.array([actor.shape.dimensions.x, actor.shape.dimensions.y, actor.shape.dimensions.z])
            fallback = np.array([.6, .6, 1.8] if label == 7 else [4.5, 2., 1.5])
            dimensions = np.where(np.isfinite(dimensions) & (dimensions > 0), dimensions, fallback)
            predictions = [path for path in actor.kinematics.predicted_paths if path.path and path.confidence > 0]
            if not predictions:
                times = np.array([0., config.horizon])
                start = np.array(position(source_pose.position))
                velocity = source_rotation.apply([actor.kinematics.initial_twist_with_covariance.twist.linear.x, 0., 0.])
                points = convert([start, start + velocity * config.horizon])
                samples = [(points, times, 1.)]
            else:
                confidence_sum = sum(path.confidence for path in predictions)
                samples = []
                for path in predictions:
                    interval = path.time_step.sec + path.time_step.nanosec * 1e-9
                    if len(path.path) > 1 and interval <= 0:
                        raise ValueError('Prediction time step must be positive')
                    if not np.isfinite(path.confidence) or not 0 < path.confidence <= 1:
                        raise ValueError('Invalid trajectory confidence')
                    points = convert([position(pose.position) for pose in path.path])
                    samples.append((points, np.arange(len(points)) * interval, path.confidence / confidence_sum))
            for points, times, confidence in samples:
                branches.append(Branch(points, times, confidence, existence, kind, tuple(dimensions),
                                       bytes(actor.object_id.uuid) in virtual_ids, heading, bytes(actor.object_id.uuid)))
                if len(branches) > min(config.maximum_branches, config.gp_support_points // 2):
                    raise ValueError('Prediction branch capacity exceeded')
    state = np.r_[translation, yaw, ego.twist.twist.linear.x,
                  ego.twist.twist.linear.y, ego.twist.twist.angular.z]
    if not np.isfinite(state).all() or state[4] < 0:
        raise ValueError('Invalid ego state')
    return Snapshot(version, stamp_ns(ego.header.stamp), monotonic(), state,
                    np.array([ego.accel.accel.linear.x,
                              ego.tire_angle_front]), road, tuple(lanes), tuple(branches), ego_frame)


def trajectory_message(plan, snapshot, config):
    from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
    message = Trajectory()
    message.header.frame_id = snapshot.frame_id
    message.header.stamp.sec = snapshot.stamp_ns // 1_000_000_000
    message.header.stamp.nanosec = snapshot.stamp_ns % 1_000_000_000
    for index, state in enumerate(plan.states):
        point = TrajectoryPoint()
        elapsed = round(index * config.control_step * 1e9)
        point.time_from_start.sec = elapsed // 1_000_000_000
        point.time_from_start.nanosec = elapsed % 1_000_000_000
        point.pose.position.x, point.pose.position.y, point.pose.position.z = map(float, state[:3])
        _, heading, slope = snapshot.road.project(state[:3])
        pitch = np.arctan(slope * np.cos(state[3] - heading))
        orientation = Rotation.from_euler('xyz', [0, -pitch, state[3]]).as_quat()
        point.pose.orientation.x, point.pose.orientation.y, point.pose.orientation.z, point.pose.orientation.w = map(float, orientation)
        point.longitudinal_velocity_mps = float(state[4])
        point.lateral_velocity_mps = float(state[5])
        point.heading_rate_rps = float(state[6])
        sample = min(index, len(plan.commands) - 1)
        point.acceleration_mps2 = float(plan.accelerations[sample, 0])
        point.front_wheel_angle_rad = float(plan.commands[sample, 1])
        message.points.append(point)
    return message