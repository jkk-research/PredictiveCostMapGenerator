"""Ordered road geometry without a y(x) polynomial assumption."""

from dataclasses import dataclass
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.prepared import prep


def resample_polyline(points, count):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('A finite road polyline with at least two points is required')
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-8]
    points, arc = points[keep], arc[keep]
    if len(points) < 2:
        raise ValueError('Degenerate road polyline')
    samples = np.linspace(0, arc[-1], count)
    return np.column_stack([np.interp(samples, arc, points[:, axis]) for axis in range(3)])


@dataclass(frozen=True)
class Road:
    points: np.ndarray
    width: float
    speed: float
    drivable_area: object = None

    def parameter_points(self, count):
        samples = resample_polyline(self.points, count)
        widths = np.full((count, 2), self.width / 2)
        if self.drivable_area is not None:
            bounds = self.drivable_area.bounds
            span = max(np.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]), self.width)
            tangents = np.gradient(samples[:, :2], axis=0)
            for index, (point, tangent) in enumerate(zip(samples, tangents)):
                normal = np.array([-tangent[1], tangent[0]]) / max(np.linalg.norm(tangent), 1e-12)
                section = LineString([point[:2] - span * normal, point[:2] + span * normal])
                intersection = self.drivable_area.intersection(section)
                segments = list(intersection.geoms) if hasattr(intersection, 'geoms') else [intersection]
                intervals = []
                for segment in segments:
                    if segment.geom_type != 'LineString' or segment.is_empty:
                        continue
                    values = (np.asarray(segment.coords)[:, :2] - point[:2]) @ normal
                    intervals.append((float(np.min(values)), float(np.max(values))))
                merged = []
                for lower, upper in sorted(intervals):
                    if merged and lower <= merged[-1][1] + 1e-6:
                        merged[-1] = (merged[-1][0], max(upper, merged[-1][1]))
                    else:
                        merged.append((lower, upper))
                for lower, upper in merged:
                    if lower <= 1e-6 and upper >= -1e-6:
                        widths[index] = [max(0., upper), max(0., -lower)]
                        break
        return np.column_stack([samples, widths])

    def contains_trajectory(self, states, length, width, margin):
        if self.drivable_area is None:
            return True
        permitted = prep(self.drivable_area.buffer(1e-6))
        corners = np.array([[-1., -1.], [-1., 1.], [1., 1.], [1., -1.]]) * [length / 2 + margin, width / 2 + margin]
        for state in states:
            cosine, sine = np.cos(state[3]), np.sin(state[3])
            rotation = np.array([[cosine, -sine], [sine, cosine]])
            footprint = Polygon(corners @ rotation.T + state[:2])
            if not permitted.covers(footprint):
                return False
        return True

    def project(self, position):
        first, delta = self.points[:-1], np.diff(self.points, axis=0)
        length_squared = np.sum(delta[:, :2]**2, axis=1)
        fraction = np.clip(np.sum((np.asarray(position)[:2] - first[:, :2]) * delta[:, :2], axis=1) /
                           np.maximum(length_squared, 1e-12), 0, 1)
        projected = first + fraction[:, None] * delta
        index = np.argmin(np.sum((projected[:, :2] - np.asarray(position)[:2])**2, axis=1))
        heading = np.arctan2(delta[index, 1], delta[index, 0])
        slope = delta[index, 2] / np.sqrt(max(length_squared[index], 1e-12))
        return projected[index], heading, slope


def footprints_overlap(first_position, first_heading, first_size,
                       second_position, second_heading, second_size, margin=0.0):
    difference = np.asarray(second_position)[:2] - np.asarray(first_position)[:2]
    first_axes = np.array([[np.cos(first_heading), np.sin(first_heading)],
                           [-np.sin(first_heading), np.cos(first_heading)]])
    second_axes = np.array([[np.cos(second_heading), np.sin(second_heading)],
                            [-np.sin(second_heading), np.cos(second_heading)]])
    for axis in np.vstack([first_axes, second_axes]):
        first_radius = np.sum(np.abs(first_axes @ axis) * np.asarray(first_size)[:2] / 2)
        second_radius = np.sum(np.abs(second_axes @ axis) * np.asarray(second_size)[:2] / 2)
        if abs(difference @ axis) > first_radius + second_radius + margin:
            return False
    return True


def longitudinal_tracking_index(states, measured, minimum_index=0):
    candidates = states[minimum_index:]
    position_error = np.sum((candidates[:, :2] - measured[:2])**2, axis=1)
    speed_error = .05 * (candidates[:, 4] - measured[4])**2
    yaw_error = .1 * np.arctan2(np.sin(candidates[:, 3] - measured[3]),
                               np.cos(candidates[:, 3] - measured[3]))**2
    return minimum_index + int(np.argmin(position_error + speed_error + yaw_error))


def new_virtual_hazard(branches, known_identifiers, states, times, ego_size, margin):
    return any(branch.virtual and branch.identifier not in known_identifiers and
               trajectory_overlaps(states, branch.sample(times), branch.headings(times),
                                   ego_size, branch.dimensions, margin) for branch in branches)


def trajectory_overlaps(states, positions, headings, first_size, second_size, margin):
    difference = positions[:, :2] - states[:, :2]
    separated = np.zeros(len(states), dtype=bool)
    for axis in (states[:, 3], states[:, 3] + np.pi / 2, headings, headings + np.pi / 2):
        distance = np.abs(difference[:, 0] * np.cos(axis) + difference[:, 1] * np.sin(axis))
        first_radius = (first_size[0] / 2 * np.abs(np.cos(states[:, 3] - axis)) +
                        first_size[1] / 2 * np.abs(np.sin(states[:, 3] - axis)))
        second_radius = (second_size[0] / 2 * np.abs(np.cos(headings - axis)) +
                         second_size[1] / 2 * np.abs(np.sin(headings - axis)))
        separated |= distance > first_radius + second_radius + margin
    return np.any(~separated)