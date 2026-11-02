import math
import queue
from dataclasses import dataclass, field

import numpy as np

from plan_predictive_cost_map.pcm_lane_geometry import fit_drivable_gp, prepare_progress_reference


@dataclass
class PredictionBranch:
    predicted_path: np.ndarray
    prediction_time_step: float
    confidence: float


@dataclass
class ObjectState:
    x: float
    y: float
    theta: float
    vx: float
    ax: float
    type: int
    # A 2D NumPy array of shape (N, 2) representing [[x1, y1], [x2, y2], ...]
    predicted_path: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    prediction_time_step: float = 0.1
    existence_probability: float = 1.0
    trajectory_confidence: float = 1.0
    predicted_branches: list = field(default_factory=list)


@dataclass(frozen=True)
class ScenarioSettings:
    prediction_time_step: float
    reference_samples: int
    target_support_points: int
    drivable_support_points: int
    target_kernel: tuple
    drivable_kernel: tuple
    target_cost: float
    drivable_cost: float
    target_noise_std: float
    drivable_noise_std: float


def quaternion_to_yaw(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def parse_objects(message, settings):
    pedestrians, vehicles = [], []
    for obj in message.local_moving_objects.objects:
        pose = obj.kinematics.initial_pose_with_covariance.pose
        x, y = pose.position.x, pose.position.y
        if not (math.sqrt(x**2 + y**2) < 150 and x >= -150):
            continue
        existence = float(obj.existence_probability)
        if not math.isfinite(existence) or existence <= 0.0:
            continue
        branches = []
        for prediction in obj.kinematics.predicted_paths:
            confidence = float(prediction.confidence)
            if not math.isfinite(confidence) or confidence <= 0.0 or not prediction.path:
                continue
            points = np.array([[point.position.x, point.position.y] for point in prediction.path])
            if not np.isfinite(points).all():
                continue
            branches.append(PredictionBranch(
                points, prediction.time_step.sec + prediction.time_step.nanosec * 1e-9, min(confidence, 1.0)))
        if obj.kinematics.predicted_paths and not branches:
            continue
        label = obj.classification[0].label
        state = ObjectState(
            x=x, y=y, theta=quaternion_to_yaw(pose.orientation),
            vx=obj.kinematics.initial_twist_with_covariance.twist.linear.x,
            ax=obj.kinematics.initial_acceleration_with_covariance.accel.linear.x, type=label,
            predicted_path=branches[0].predicted_path if branches else np.empty((0, 2)),
            prediction_time_step=branches[0].prediction_time_step if branches else settings.prediction_time_step,
            existence_probability=min(existence, 1.0),
            trajectory_confidence=branches[0].confidence if branches else 1.0,
            predicted_branches=branches)
        (pedestrians if label == 7 else vehicles).append(state)
    return pedestrians, vehicles


def parse_scenario(message, settings):
    """Convert a Scenario message into the planner's plain-data scenario update."""
    pedestrians, vehicles = parse_objects(message, settings)
    update = {
        'stamp_ns': message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec,
        'pedestrians': pedestrians, 'vehicles': vehicles,
        'lanes': {'ego_lane_x': [], 'ego_lane_y': [], 'ego_lane_z': [], 'left_lane_x': [], 'left_lane_y': []},
        'road_bounds': [], 'target_speed': 0.0, 'lane_received': False,
        'reference_data': None, 'target_gp': None, 'warnings': []}
    drivable_sets = []
    for index, path_with_rule in enumerate(message.paths):
        positions = np.array([[point.point.pose.position.x, point.point.pose.position.y, point.point.pose.position.z]
                              for point in path_with_rule.path.points]).reshape(-1, 3)
        geometry = positions[np.isfinite(positions).all(axis=1)]
        if index > 0:
            if len(geometry):
                drivable_sets.append(geometry[:, :2])
            continue
        for boundary in (path_with_rule.path.left_bound, path_with_rule.path.right_bound):
            samples = [[point.x, point.y, point.z] for point in boundary
                       if all(math.isfinite(value) for value in (point.x, point.y, point.z))]
            if samples:
                indices = np.linspace(0, len(samples) - 1, min(len(samples), 128), dtype=int)
                update['road_bounds'].extend(samples[sample] for sample in indices)
        update['lanes']['ego_lane_x'] = geometry[:, 0].tolist()
        update['lanes']['ego_lane_y'] = geometry[:, 1].tolist()
        update['lanes']['ego_lane_z'] = geometry[:, 2].tolist()
        if len(geometry) < 2:
            continue
        target_speed = float(path_with_rule.traffic_rules.maximum_speed)
        if not math.isfinite(target_speed) or target_speed < 0.0:
            update['warnings'].append('Ignoring ego path with invalid maximum_speed')
            continue
        update['target_speed'] = target_speed
        try:
            update['reference_data'] = prepare_progress_reference(geometry, settings.reference_samples)
        except ValueError as error:
            update['warnings'].append(f'Invalid target reference: {error}')
            continue
        update['target_gp'] = fit_drivable_gp(
            geometry[:, :2], settings.target_support_points, settings.target_kernel,
            settings.target_cost, settings.target_noise_std)
        update['lane_received'] = True
    merged_drivable = np.vstack(drivable_sets) if drivable_sets else np.empty((0, 2))
    update['drivable_gp'] = fit_drivable_gp(
        merged_drivable, settings.drivable_support_points, settings.drivable_kernel,
        settings.drivable_cost, settings.drivable_noise_std)
    return update


def scenario_worker(requests, results, settings):
    """Deserialize and preprocess the newest serialized Scenario, dropping stale backlog."""
    from rclpy.serialization import deserialize_message
    from crp_msgs.msg import Scenario

    while True:
        data = requests.get()
        while data is not None:
            try:
                data = requests.get_nowait()
            except queue.Empty:
                break
            if data is None:
                return
        if data is None:
            return
        try:
            results.put(parse_scenario(deserialize_message(data, Scenario), settings))
        except Exception as error:  # Report to the planner instead of silently stopping updates.
            results.put({'error': repr(error)})
