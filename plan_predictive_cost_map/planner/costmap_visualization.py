"""Fixed-scale, raw-valued time slices of the GP cost field."""

import numpy as np


COST_FIELDS = ('x', 'y', 'z', 'cost', 'vehicle', 'pedestrian', 'cyclist', 'obstacle', 'target', 'road', 'risk_limit', 'prediction_time')
POINT_DTYPE = np.dtype([(name, '<f4') for name in COST_FIELDS] + [('rgb', '<u4')])


def cost_colors(cost):
    stops = np.array([-1., -.25, .05, 1., 3.])
    palette = np.array([[25., 90., 190.], [30., 190., 170.], [240., 200., 35.],
                        [235., 100., 30.], [180., 30., 50.]])
    channels = [np.interp(cost, stops, palette[:, axis]).astype(np.uint32) for axis in range(3)]
    return channels[0] << 16 | channels[1] << 8 | channels[2]


def costmap_clouds(costmap, snapshot, config, times=(0., 1., 2., 3.), resolution=1.,
                   forward=80., backward=5., width=30.):
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    if resolution <= 0 or min(forward, backward, width) <= 0:
        raise ValueError('Cost-map display dimensions must be positive')
    grid_x, grid_y = np.meshgrid(np.arange(-backward, forward + .5 * resolution, resolution),
                                 np.arange(-width / 2, width / 2 + .5 * resolution, resolution))
    local = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    yaw = snapshot.state[3]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    points = local @ rotation.T + snapshot.state[:2]
    route = snapshot.road.points
    delta = np.diff(route, axis=0)
    difference = points[:, None, :] - route[None, :-1, :2]
    squared = np.maximum(np.sum(delta[:, :2]**2, axis=1), 1e-12)
    fraction = np.clip(np.sum(difference * delta[None, :, :2], axis=2) / squared, 0, 1)
    projected_xy = route[None, :-1, :2] + fraction[:, :, None] * delta[None, :, :2]
    nearest = np.argmin(np.sum((points[:, None, :] - projected_xy)**2, axis=2), axis=1)
    height = route[nearest, 2] + fraction[np.arange(len(points)), nearest] * delta[nearest, 2] + .15
    header = Header(frame_id='map')
    header.stamp.sec = snapshot.stamp_ns // 1_000_000_000
    header.stamp.nanosec = snapshot.stamp_ns % 1_000_000_000
    clouds = []
    for elapsed in times:
        values = costmap.layers(np.column_stack([points, np.full(len(points), elapsed)]), config)
        combined = np.maximum.reduce([*values[:4], np.minimum(values[4], values[5])])
        data = np.zeros(len(points), dtype=POINT_DTYPE)
        data['x'], data['y'], data['z'] = points[:, 0], points[:, 1], height
        data['cost'], data['risk_limit'], data['rgb'] = combined, config.risk_limit, cost_colors(combined)
        data['prediction_time'] = elapsed
        for index, name in enumerate(('vehicle', 'pedestrian', 'cyclist', 'obstacle', 'target', 'road')):
            data[name] = values[index]
        cloud = PointCloud2(header=header, height=1, width=len(points), is_bigendian=False,
                            point_step=POINT_DTYPE.itemsize, row_step=POINT_DTYPE.itemsize * len(points),
                            is_dense=bool(np.isfinite(combined).all()), data=data.tobytes())
        cloud.fields = [PointField(name=name, offset=POINT_DTYPE.fields[name][1],
                                  datatype=PointField.UINT32 if name == 'rgb' else PointField.FLOAT32, count=1)
                        for name in POINT_DTYPE.names]
        clouds.append(cloud)
    return clouds