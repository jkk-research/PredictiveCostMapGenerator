import json
import math
from timeit import default_timer

import casadi as ca
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker

from plan_predictive_cost_map.planner.costmap_visualization import cost_colors
from plan_predictive_cost_map.pcm_kernels import lane_kernel, rbf_kernel


def evaluate_field(snapshot, queries, batch_size=256):
    queries = np.asarray(queries, dtype=float)
    values = {}
    for name, layer in snapshot['layers'].items():
        support = ca.DM(layer['points'])
        kernel = np.asarray(layer['kernel'], dtype=float)
        if name in ('ego_lane', 'opposite_lane'):
            alpha = ca.DM(layer['alpha'])
            values[name] = np.concatenate([
                (lane_kernel(ca.DM(queries[start:start + batch_size]), support, kernel)
                 @ alpha + layer['offset']).full().flatten()
                for start in range(0, len(queries), batch_size)
            ])
            continue
        alpha = ca.DM(layer['alpha'])
        pieces = []
        for start in range(0, len(queries), batch_size):
            query = ca.DM(queries[start:start + batch_size])
            mean = rbf_kernel(query, support, kernel) @ alpha + layer['offset']
            pieces.append(mean.full().flatten())
        values[name] = np.concatenate(pieces)
    def smooth_max(left, right):
        return 0.5 * (left + right + np.sqrt((left - right)**2 + 0.01))
    dynamic = smooth_max(values['pedestrian'], values['vehicle'])
    lanes = np.minimum(values['ego_lane'], values['opposite_lane'])
    return smooth_max(dynamic, lanes)


def road_height(points, road, road_bounds=()):
    road = np.asarray(road, dtype=float)
    if len(road) < 2:
        raise ValueError('At least two road points are required')
    heights = []
    delta = np.diff(road, axis=0)
    squared_length = np.maximum(np.sum(delta[:, :2]**2, axis=1), 1e-12)
    for start in range(0, len(points), 256):
        query = points[start:start + 256]
        difference = query[:, None, :] - road[None, :-1, :2]
        fraction = np.clip(np.sum(difference * delta[None, :, :2], axis=2) / squared_length, 0., 1.)
        projection = road[None, :-1, :2] + fraction[:, :, None] * delta[None, :, :2]
        nearest = np.argmin(np.sum((query[:, None, :] - projection)**2, axis=2), axis=1)
        heights.append(road[nearest, 2] + fraction[np.arange(len(query)), nearest] * delta[nearest, 2])
    heights = np.concatenate(heights)
    if len(road_bounds) >= 3:
        surface = np.vstack((road, np.asarray(road_bounds, dtype=float)))
        try:
            interpolated = LinearNDInterpolator(surface[:, :2], surface[:, 2])(points)
            valid = np.isfinite(interpolated)
            heights[valid] = interpolated[valid]
        except QhullError:
            pass
    return heights


def surface_messages(snapshot, grid_x, grid_y, prediction_time, height_scale, surface_offset,
                     opacity, lifetime):
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    query = np.column_stack((points, np.full(len(points), prediction_time)))
    means = evaluate_field(snapshot, query)
    base_height = road_height(points, snapshot['road'], snapshot.get('road_bounds', ()))
    heights = base_height + surface_offset + height_scale * means
    finite = np.isfinite(means) & np.isfinite(heights)
    rgb = cost_colors(np.where(finite, means, 0.))
    marker = Marker()
    marker.header.frame_id = snapshot['frame_id']
    marker.header.stamp.sec = snapshot['stamp_ns'] // 1_000_000_000
    marker.header.stamp.nanosec = snapshot['stamp_ns'] % 1_000_000_000
    marker.ns = 'pcm_gp_mean'
    marker.id = 0
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = marker.scale.y = marker.scale.z = 1.0
    marker.color.a = opacity
    marker.lifetime.sec = int(lifetime)
    marker.lifetime.nanosec = int((lifetime - int(lifetime)) * 1e9)
    rows, columns = grid_x.shape
    indices = np.arange(rows * columns).reshape(rows, columns)
    lower_left = indices[:-1, :-1].ravel()
    triangles = np.concatenate((np.column_stack((lower_left, lower_left + 1, lower_left + columns)),
                                np.column_stack((lower_left + 1, lower_left + columns + 1,
                                                 lower_left + columns))))
    triangles = triangles[finite[triangles].all(axis=1)]
    vertices = [Point(x=float(point[0]), y=float(point[1]), z=float(height))
                for point, height in zip(points, heights)]
    colors = [ColorRGBA(r=float((color >> 16) & 255) / 255.,
                        g=float((color >> 8) & 255) / 255., b=float(color & 255) / 255., a=opacity)
              for color in rgb]
    marker.points = [vertices[index] for index in triangles.ravel()]
    marker.colors = [colors[index] for index in triangles.ravel()]
    dtype = np.dtype([(name, '<f4') for name in ('x', 'y', 'z', 'mean', 'road_z', 'prediction_time')]
                     + [('rgb', '<u4')])
    data = np.zeros(len(points), dtype=dtype)
    data['x'], data['y'], data['z'] = points[:, 0], points[:, 1], heights
    data['mean'], data['road_z'], data['prediction_time'], data['rgb'] = means, base_height, prediction_time, rgb
    cloud = PointCloud2(header=marker.header, height=rows, width=columns, is_bigendian=False,
                        point_step=dtype.itemsize, row_step=dtype.itemsize * columns,
                        is_dense=bool(finite.all()), data=data.tobytes())
    cloud.fields = [PointField(name=name, offset=dtype.fields[name][1], count=1,
                              datatype=PointField.UINT32 if name == 'rgb' else PointField.FLOAT32)
                    for name in dtype.names]
    return marker, cloud


def validate_snapshot(snapshot):
    if snapshot['version'] != 2 or snapshot['frame_id'] != 'base_link':
        raise ValueError('Expected a version 2 base_link GP snapshot')
    if not isinstance(snapshot['stamp_ns'], int) or snapshot['stamp_ns'] < 0:
        raise ValueError('Invalid snapshot timestamp')
    road = np.asarray(snapshot['road'], dtype=float)
    if road.ndim != 2 or road.shape[1] != 3 or len(road) < 2 or not np.isfinite(road).all():
        raise ValueError('Invalid road geometry')
    bounds = np.asarray(snapshot.get('road_bounds', []), dtype=float)
    if bounds.size and (bounds.ndim != 2 or bounds.shape[1] != 3 or not np.isfinite(bounds).all()):
        raise ValueError('Invalid road boundary geometry')
    times = np.asarray(snapshot['times'], dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0.):
        raise ValueError('Invalid prediction times')
    for name in ('pedestrian', 'vehicle', 'ego_lane', 'opposite_lane'):
        layer = snapshot['layers'][name]
        points, alpha, kernel = (np.asarray(layer[key], dtype=float) for key in ('points', 'alpha', 'kernel'))
        dimensions = 2 if name in ('ego_lane', 'opposite_lane') else 3
        if (points.ndim != 2 or points.shape[1] != dimensions or len(points) == 0
            or alpha.shape != (len(points),) or kernel.shape != (dimensions + 1,)
                or not all(np.isfinite(value).all() for value in (points, alpha, kernel))
                or np.any(kernel <= 0.) or not math.isfinite(layer['offset'])):
            raise ValueError(f'Invalid {name} GP layer')


class CostmapViewer(Node):
    def __init__(self):
        super().__init__('pcm_costmap_viewer')
        defaults = {'resolution': 1.0, 'forward': 60.0, 'backward': 5.0, 'width': 20.0,
                    'update_rate': 4.0, 'prediction_time': 0.5, 'height_scale': 1.0,
                    'surface_offset': 1.15, 'opacity': 0.65, 'stale_timeout': 2.0,
                    'max_grid_points': 20000, 'snap_to_prediction_time': True}
        self.settings = {name: self.declare_parameter(name, value).value for name, value in defaults.items()}
        for name in ('resolution', 'forward', 'backward', 'width', 'update_rate', 'stale_timeout'):
            if not math.isfinite(self.settings[name]) or self.settings[name] <= 0.:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('prediction_time', 'height_scale', 'surface_offset'):
            if not math.isfinite(self.settings[name]) or self.settings[name] < 0.:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not 0. < self.settings['opacity'] <= 1. or self.settings['max_grid_points'] < 4:
            raise ValueError('Invalid opacity or grid budget')
        resolution = self.settings['resolution']
        columns = math.ceil((self.settings['forward'] + self.settings['backward']) / resolution) + 1
        rows = math.ceil(self.settings['width'] / resolution) + 1
        if rows * columns > self.settings['max_grid_points']:
            raise ValueError('Requested grid exceeds max_grid_points')
        self.grid_x, self.grid_y = np.meshgrid(
            np.linspace(-self.settings['backward'], self.settings['forward'], columns),
            np.linspace(-self.settings['width'] / 2., self.settings['width'] / 2., rows))
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.surface_pub = self.create_publisher(Marker, '/plan/pcm_longitudinal/cost_surface', 1)
        self.cloud_pub = self.create_publisher(PointCloud2, '/plan/pcm_longitudinal/cost_cloud', 1)
        self.subscription = self.create_subscription(String, '/plan/pcm_longitudinal/gp_state', self.receive, qos)
        self.snapshot = None
        self.timer = self.create_timer(1. / self.settings['update_rate'], self.render)

    def receive(self, message):
        try:
            snapshot = json.loads(message.data)
            validate_snapshot(snapshot)
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            self.get_logger().warn(f'Ignoring malformed GP snapshot: {error}')
            return
        self.snapshot = snapshot
        self.received_at = default_timer()

    def render(self):
        if self.snapshot is None or default_timer() - self.received_at > self.settings['stale_timeout']:
            return
        snapshot, self.snapshot = self.snapshot, None
        requested = self.settings['prediction_time']
        if requested > snapshot['times'][-1]:
            self.get_logger().warn('prediction_time exceeds the planner horizon; skipping surface')
            return
        prediction_time = requested
        if self.settings['snap_to_prediction_time']:
            prediction_time = min(snapshot['times'], key=lambda value: abs(value - requested))
        marker, cloud = surface_messages(snapshot, self.grid_x, self.grid_y, prediction_time,
                                          self.settings['height_scale'], self.settings['surface_offset'],
                                          self.settings['opacity'], self.settings['stale_timeout'])
        if not cloud.is_dense:
            self.get_logger().warn('Nonfinite GP values omitted from the cost surface')
        self.surface_pub.publish(marker)
        self.cloud_pub.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = CostmapViewer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()