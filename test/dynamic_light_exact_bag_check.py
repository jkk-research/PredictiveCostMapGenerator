import json
import sys

import casadi as ca
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from plan_predictive_cost_map.pcm_lane_geometry import progress_reference_function
from plan_predictive_cost_map.pcm_longitudinal_dynamics import build_dynamic_light_rollout


def stamp_ns(message):
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


def curvature(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.0
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, distances > 1e-8]]
    if len(points) < 3:
        return 0.0
    progress = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    heading = np.unwrap(np.arctan2(np.gradient(points[:, 1]), np.gradient(points[:, 0])))
    return float(np.max(np.abs(np.gradient(heading, progress))))


def main(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    types = {topic.name: get_message(topic.type) for topic in reader.get_all_topics_and_types()}
    messages = {'/ego': [], '/plan/trajectory': [], '/plan/pcm_longitudinal/gp_state': []}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in messages:
            messages[topic].append(deserialize_message(data, types[topic]))
    print('counts:', {key: len(value) for key, value in messages.items()})
    evaluate_reference = progress_reference_function(96)
    matched = 0
    for trajectory in messages['/plan/trajectory']:
        if not trajectory.points or not messages['/plan/pcm_longitudinal/gp_state']:
            continue
        snapshot_message = min(messages['/plan/pcm_longitudinal/gp_state'],
            key=lambda item: abs(json.loads(item.data)['stamp_ns'] - stamp_ns(trajectory)))
        snapshot = json.loads(snapshot_message.data)
        planning = snapshot.get('planning', {})
        if planning.get('prediction_model') != 'dynamic_light' or not planning.get('accepted'):
            continue
        if 'progress_reference' not in snapshot or 'optimized_coefficients' not in snapshot:
            raise RuntimeError('Missing exact reference or coefficients in GP snapshot')
        planned = np.asarray([[point.pose.position.x, point.pose.position.y] for point in trajectory.points])
        plan_length = float(np.sum(np.linalg.norm(np.diff(planned, axis=0), axis=1)))
        reference = ca.DM(snapshot['progress_reference'])
        progress = float(reference[0]) + np.linspace(0.0, plan_length, 40)
        reference_points = np.asarray([evaluate_reference(value, reference).full().flatten()[:3]
                                       for value in progress])
        reference_curve = curvature(reference_points[:, :2])
        plan_curve = curvature(planned)
        if reference_curve < 0.05 or plan_curve > 0.01:
            continue
        ego = min(messages['/ego'], key=lambda item: abs(stamp_ns(item) - stamp_ns(trajectory)))
        pose = ego.pose.pose
        q = pose.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
        pitch = -np.arcsin(np.clip(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0))
        initial = ca.DM([pose.position.x, pose.position.y, pose.position.z, yaw,
                         ego.twist.twist.linear.x, ego.twist.twist.linear.y, ego.twist.twist.angular.z])
        horizon = float(trajectory.points[-1].time_from_start.sec +
                        trajectory.points[-1].time_from_start.nanosec * 1e-9)
        rollout = build_dynamic_light_rollout(horizon, len(trajectory.points), 0.1, min(0.25, horizon))
        direct = rollout(np.asarray(snapshot['optimized_coefficients']), initial,
                         ego.accel.accel.linear.x, ego.tire_angle_front, np.zeros(4), np.zeros(4),
                         pitch, reference)
        direct_points = direct[0].full()[:, :2]
        print(json.dumps({
            'stamp_ns': stamp_ns(trajectory),
            'gp_age_ms': round(abs(snapshot['stamp_ns'] - stamp_ns(trajectory)) / 1e6, 2),
            'reference_curve_in_preview': reference_curve,
            'published_plan_curve': plan_curve,
            'direct_rollout_curve': curvature(direct_points),
            'direct_to_published_max_m': float(np.max(np.linalg.norm(direct_points - planned, axis=1))),
            'optimizer_status': planning.get('status'),
            'optimizer_iterations': planning.get('iterations'),
        }, sort_keys=True))
        matched += 1
        if matched >= 12:
            break
    print('matching_curved_samples:', matched)


if __name__ == '__main__':
    main(sys.argv[1])import json
import sys

import casadi as ca
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from plan_predictive_cost_map.pcm_lane_geometry import progress_reference_function
from plan_predictive_cost_map.pcm_longitudinal_dynamics import build_dynamic_light_rollout


def stamp_ns(message):
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


def curvature(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.0
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, distances > 1e-8]
    points = points[keep]
    if len(points) < 3:
        return 0.0
    progress = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    heading = np.unwrap(np.arctan2(np.gradient(points[:, 1]), np.gradient(points[:, 0])))
    return float(np.max(np.abs(np.gradient(heading, progress))))


def main(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    types = {topic.name: get_message(topic.type) for topic in reader.get_all_topics_and_types()}
    messages = {'/ego': [], '/plan/trajectory': [], '/plan/pcm_longitudinal/gp_state': []}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in messages:
            messages[topic].append(deserialize_message(data, types[topic]))
    print('counts:', {key: len(value) for key, value in messages.items()})
    exact_reference = progress_reference_function(96)
    matched = 0
    for trajectory in messages['/plan/trajectory']:
        snapshot_message = min(messages['/plan/pcm_longitudinal/gp_state'],
            key=lambda item: abs(json.loads(item.data)['stamp_ns'] - stamp_ns(trajectory)))
        snapshot = json.loads(snapshot_message.data)
        planning = snapshot.get('planning', {})
        if planning.get('prediction_model') != 'dynamic_light' or not planning.get('accepted'):
            continue
        if 'progress_reference' not in snapshot or 'optimized_coefficients' not in snapshot:
            raise RuntimeError('Bag GP state lacks exact progress reference or coefficients')
        planned = np.asarray([[point.pose.position.x, point.pose.position.y] for point in trajectory.points])
        plan_length = float(np.sum(np.linalg.norm(np.diff(planned, axis=0), axis=1)))
        reference = ca.DM(snapshot['progress_reference'])
        progress = float(reference[0]) + np.linspace(0., plan_length, 40)
        reference_points = np.asarray([exact_reference(value, reference).full().flatten()[:3]
                                       for value in progress])
        reference_curve = curvature(reference_points[:, :2])
        plan_curve = curvature(planned)
        if reference_curve < 0.05:
            continue
        ego = min(messages['/ego'], key=lambda item: abs(stamp_ns(item) - stamp_ns(trajectory)))
        pose = ego.pose.pose
        q = pose.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
        pitch = -np.arcsin(np.clip(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0))
        initial = ca.DM([pose.position.x, pose.position.y, pose.position.z, yaw,
                         ego.twist.twist.linear.x, ego.twist.twist.linear.y, ego.twist.twist.angular.z])
        horizon = float(trajectory.points[-1].time_from_start.sec +
                        trajectory.points[-1].time_from_start.nanosec * 1e-9)
        rollout = build_dynamic_light_rollout(horizon, len(trajectory.points), 0.1, min(0.25, horizon))
        direct = rollout(np.asarray(snapshot['optimized_coefficients']), initial,
                         ego.accel.accel.linear.x, ego.tire_angle_front, np.zeros(4), np.zeros(4),
                         pitch, reference)
        direct_xy = direct[0].full()[:, :2]
        print(json.dumps({
            'stamp_ns': stamp_ns(trajectory),
            'gp_age_ms': round(abs(snapshot['stamp_ns'] - stamp_ns(trajectory)) / 1e6, 2),
            'reference_curve_in_preview': reference_curve,
            'published_plan_curve': plan_curve,
            'direct_rollout_curve': curvature(direct_xy),
            'direct_to_published_max_m': float(np.max(np.linalg.norm(direct_xy - planned, axis=1))),
            'model': planning.get('prediction_model'),
            'accepted': planning.get('accepted'),
            'solver': planning.get('status'),
            'iterations': planning.get('iterations'),
        }, sort_keys=True))
        matched += 1
        if matched >= 12:
            break
    print('curved accepted plans:', matched)


if __name__ == '__main__':
    main(sys.argv[1])import json
import sys

import casadi as ca
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from plan_predictive_cost_map.pcm_lane_geometry import progress_reference_function
from plan_predictive_cost_map.pcm_longitudinal_dynamics import build_dynamic_light_rollout


def stamp_ns(message):
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


def curvature(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.0
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, distances > 1e-8]
    points = points[keep]
    if len(points) < 3:
        return 0.0
    progress = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    heading = np.unwrap(np.arctan2(np.gradient(points[:, 1]), np.gradient(points[:, 0])))
    return float(np.max(np.abs(np.gradient(heading, progress))))


def main(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    types = {topic.name: get_message(topic.type) for topic in reader.get_all_topics_and_types()}
    messages = {'/ego': [], '/plan/trajectory': [], '/plan/pcm_longitudinal/gp_state': []}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in messages:
            messages[topic].append(deserialize_message(data, types[topic]))
    print('message counts:', {key: len(value) for key, value in messages.items()})
    if messages['/ego']:
        ego_first, ego_last = messages['/ego'][0], messages['/ego'][-1]
        print('ego first/last:',
              [ego_first.pose.pose.position.x, ego_first.pose.pose.position.y],
              [ego_last.pose.pose.position.x, ego_last.pose.pose.position.y],
              'speed', ego_first.twist.twist.linear.x, ego_last.twist.twist.linear.x)
    if messages['/plan/pcm_longitudinal/gp_state']:
        first_snapshot = json.loads(messages['/plan/pcm_longitudinal/gp_state'][0].data)
        latest_snapshot = json.loads(messages['/plan/pcm_longitudinal/gp_state'][-1].data)
        print('snapshot road first/last:', first_snapshot['road'][0], first_snapshot['road'][-1],
              latest_snapshot['road'][0], latest_snapshot['road'][-1])
    exact_reference = progress_reference_function(96)
    count = 0
    examined = 0
    for trajectory in messages['/plan/trajectory']:
        if not trajectory.points:
            continue
        snapshot_message = min(messages['/plan/pcm_longitudinal/gp_state'],
            key=lambda item: abs(json.loads(item.data)['stamp_ns'] - stamp_ns(trajectory)))
        snapshot = json.loads(snapshot_message.data)
        if 'progress_reference' not in snapshot or 'optimized_coefficients' not in snapshot:
            raise RuntimeError('The GP-state message has no exact progress reference/coefficient fields')
        planning = snapshot.get('planning', {})
        if planning.get('prediction_model') != 'dynamic_light' or not planning.get('accepted'):
            continue

        planned = np.asarray([[point.pose.position.x, point.pose.position.y] for point in trajectory.points])
        planned_length = float(np.sum(np.linalg.norm(np.diff(planned, axis=0), axis=1)))
        reference = ca.DM(snapshot['progress_reference'])
        progress = float(reference[0]) + np.linspace(0.0, max(planned_length, 1.0), 40)
        reference_points = np.asarray([exact_reference(value, reference).full().flatten()[:3]
                                       for value in progress])
        reference_curvature = curvature(reference_points[:, :2])
        plan_curvature = curvature(planned)
        if examined < 12:
            print('SAMPLE', json.dumps({
                'stamp_ns': stamp_ns(trajectory),
                'model': planning.get('prediction_model'),
                'accepted': planning.get('accepted'),
                'reference_curvature': reference_curvature,
                'plan_curvature': plan_curvature,
                'plan_length': planned_length,
            }, sort_keys=True))
            examined += 1
        if reference_curvature < 0.05 or plan_curvature > 0.01:
            continue

        times = np.asarray([point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
                            for point in trajectory.points])
        horizon = float(times[-1])
        prediction_steps = len(trajectory.points)
        ego = min(messages['/ego'], key=lambda item: abs(stamp_ns(item) - stamp_ns(trajectory)))
        pose = ego.pose.pose
        quaternion = pose.orientation
        yaw = np.arctan2(2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
                 1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2))
        pitch = -np.arcsin(np.clip(2.0 * (quaternion.w * quaternion.y -
                         quaternion.z * quaternion.x), -1.0, 1.0))
        initial = ca.DM([pose.position.x, pose.position.y, pose.position.z, yaw,
                 ego.twist.twist.linear.x, ego.twist.twist.linear.y,
                 ego.twist.twist.angular.z])
        rollout = build_dynamic_light_rollout(horizon, prediction_steps, 0.1, min(0.25, horizon))
        direct = rollout(np.asarray(snapshot['optimized_coefficients']), initial,
                 ego.accel.accel.linear.x, ego.tire_angle_front,
                 ca.DM.zeros(4), ca.DM.zeros(4), pitch, reference)
        direct_points = direct[0].full()[:, :2]
        print(json.dumps({
            'stamp_ns': stamp_ns(trajectory),
            'gp_stamp_age_ms': round(abs(snapshot['stamp_ns'] - stamp_ns(trajectory)) / 1e6, 2),
            'ego_speed_mps': ego.twist.twist.linear.x,
            'reference_curvature_in_plan_distance': reference_curvature,
            'plan_curvature': plan_curvature,
            'direct_rollout_curvature': curvature(direct_points),
            'direct_rollout_to_published_plan_max_m': float(np.max(np.linalg.norm(direct_points - planned, axis=1))),
            'optimizer_status': planning.get('status'),
            'optimizer_iterations': planning.get('iterations'),
            'optimizer_coefficients': snapshot['optimized_coefficients'],
        }, sort_keys=True))
        count += 1
        if count == 10:
            break
    print('matching_curved_samples:', count)


if __name__ == '__main__':
    main(sys.argv[1])