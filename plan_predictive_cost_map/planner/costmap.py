"""Bounded Gaussian-process posterior fitting with MATLAB probability semantics."""

from dataclasses import dataclass
import casadi as ca
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .config import PlannerConfig
from .geometry import resample_polyline


@dataclass(frozen=True)
class Branch:
    points: np.ndarray
    times: np.ndarray
    confidence: float
    existence: float
    kind: int
    dimensions: tuple
    virtual: bool = False
    heading: float = 0.0
    identifier: bytes = b''

    def sample(self, times):
        return np.column_stack([np.interp(times, self.times, self.points[:, axis]) for axis in range(3)])

    def headings(self, times):
        positions = self.sample(times)
        delta = np.diff(positions[:, :2], axis=0)
        result = np.where(np.linalg.norm(delta, axis=1) > 1e-6,
                          np.arctan2(delta[:, 1], delta[:, 0]), self.heading)
        return np.r_[result, result[-1]] if len(result) else np.full(len(times), self.heading)


def kernel(first, second, scales, amplitude=1.0):
    difference = (np.asarray(first)[:, None, :] - np.asarray(second)[None, :, :]) / scales
    return amplitude**2 * np.exp(-0.5 * np.sum(difference**2, axis=2))


def fit_posterior(points, targets, noise, scales, amplitude=1.0):
    covariance = kernel(points, points, scales, amplitude)
    covariance.flat[::len(points) + 1] += np.asarray(noise) + 1e-8
    return cho_solve(cho_factor(covariance, lower=True, check_finite=True),
                     np.asarray(targets), check_finite=True)


def group_scales(config):
    return np.array([
        [config.vehicle_length_scale, config.vehicle_width_scale, config.time_length_scale],
        [config.pedestrian_length_scale, config.pedestrian_width_scale, config.time_length_scale],
        [config.vehicle_length_scale, config.vehicle_width_scale, config.time_length_scale],
        [config.vehicle_length_scale, config.vehicle_width_scale, config.time_length_scale],
        [config.route_length_scale, config.route_width_scale, 1.0],
        [config.road_length_scale, config.road_width_scale, 1.0],
    ])


@dataclass(frozen=True)
class CostMap:
    centers: np.ndarray
    coefficients: np.ndarray
    omitted_branches: int = 0

    def layers(self, points, config=PlannerConfig()):
        points = np.asarray(points).reshape(-1, 3)
        values = []
        for group, scales in enumerate(group_scales(config)):
            queries = points.copy()
            if group >= 4:
                queries[:, 2] = 0
            values.append(kernel(queries, self.centers[group], scales, config.gp_amplitude) @
                          self.coefficients[group] + (-1 if group < 4 else 0))
        return np.vstack(values)

    def evaluate(self, points, config=PlannerConfig()):
        values = self.layers(points, config)
        return np.maximum.reduce([*values[:4], np.minimum(values[4], values[5])])


def fit_costmap(branches, route, lanes, origin, config=PlannerConfig()):
    support = config.gp_support_points
    centers = np.zeros((6, support, 3))
    coefficients = np.zeros((6, support))
    candidates = [branch for branch in branches if branch.existence > 0 and branch.confidence > 0]
    candidates.sort(key=lambda branch: np.min(np.linalg.norm(branch.points[:, :2] - np.asarray(origin)[:2], axis=1)))
    capacity = min(config.maximum_branches, support // 2)
    omitted = max(0, len(candidates) - capacity)
    candidates = candidates[:capacity]
    scales = group_scales(config)
    severities = [config.vehicle_severity, config.pedestrian_severity,
                  config.cyclist_severity, config.obstacle_severity]
    for group in range(6):
        if group < 4:
            members = [branch for branch in candidates if branch.kind == group]
            if not members:
                continue
            allocations = np.full(len(members), support // len(members), dtype=int)
            allocations[:support % len(members)] += 1
            samples, targets, noise = [], [], []
            for branch, count in zip(members, allocations):
                if count == 0:
                    continue
                times = np.linspace(0.0, config.horizon, count)
                positions = branch.sample(times)
                samples.extend(np.column_stack([positions[:, :2], times]))
                severity = severities[group]
                targets.extend(np.full(count, 1 + branch.confidence * severity))
                joint = branch.existence * branch.confidence
                initial_noise, growth = (0.02, 0.25) if group == 1 else (0.01, 0.2 if group == 2 else 0.05)
                if group == 3:
                    noise.extend(np.full(count, 0.001))
                else:
                    noise.extend(initial_noise + growth * times + joint * (1 - joint) * severity**2)
            samples = np.asarray(samples)
        else:
            roads = [route] if group == 4 else (lanes or [route])
            pieces = []
            for road in roads:
                pieces.extend(resample_polyline(road, max(2, support // len(roads))))
            samples = np.asarray(pieces)
            if len(samples) > support:
                samples = samples[np.linspace(0, len(samples) - 1, support).astype(int)]
            samples[:, 2] = 0
            targets = np.full(len(samples), config.target_cost if group == 4 else config.road_cost)
            noise = np.full(len(samples), 0.001)
        count = len(samples)
        centers[group, :count] = samples
        coefficients[group, :count] = fit_posterior(samples, targets, noise, scales[group], config.gp_amplitude)
    return CostMap(centers, coefficients, omitted)


def symbolic_cost(config):
    query = ca.SX.sym('query', 3)
    centers = ca.SX.sym('centers', 3, 6 * config.gp_support_points)
    coefficients = ca.SX.sym('coefficients', 6 * config.gp_support_points)
    values = []
    for group, scales in enumerate(group_scales(config)):
        point = ca.vertcat(query[0], query[1], 0 if group >= 4 else query[2])
        value = -1 if group < 4 else 0
        for index in range(config.gp_support_points):
            column = group * config.gp_support_points + index
            difference = (point - centers[:, column]) / scales
            value += coefficients[column] * config.gp_amplitude**2 * ca.exp(-0.5 * ca.dot(difference, difference))
        values.append(value)
    result = ca.fmin(values[4], values[5])
    for value in values[:4]:
        result = ca.fmax(result, value)
    return ca.Function('gp_cost', [query, centers, coefficients], [result])