from functools import lru_cache

import casadi as ca
import numpy as np
from scipy.interpolate import make_interp_spline

from plan_predictive_cost_map.pcm_kernels import numeric_rbf_kernel


def prepare_progress_reference(waypoints, sample_count):
    points = np.asarray(waypoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Reference needs at least two finite ordered XYZ waypoints')
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]]
    if len(points) < 2:
        raise ValueError('Reference waypoints must contain nonzero travel distance')
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    distance = np.r_[0.0, np.cumsum(lengths)]
    fraction = np.clip(np.sum(-points[:-1] * delta, axis=1) / lengths**2, 0.0, 1.0)
    projections = points[:-1] + fraction[:, None] * delta
    nearest = int(np.argmin(np.sum(projections**2, axis=1)))
    origin_progress = distance[nearest] + fraction[nearest] * lengths[nearest]
    samples = np.linspace(0.0, distance[-1], sample_count)
    coordinates = np.column_stack([np.interp(samples, distance, points[:, axis]) for axis in range(3)])
    tangent = np.column_stack([np.gradient(coordinates[:, axis], samples) for axis in range(3)])
    heading = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    pitch = np.arctan2(tangent[:, 2], np.maximum(np.linalg.norm(tangent[:, :2], axis=1), 1e-8))
    curvature = np.gradient(heading, samples)
    values = np.column_stack((coordinates, heading, pitch, curvature))
    spline = make_interp_spline(np.linspace(0.0, 1.0, sample_count), values, k=min(3, sample_count - 1))
    return np.r_[-origin_progress, distance[-1] - origin_progress, spline.c.reshape(-1)]


@lru_cache(maxsize=None)
def spline_basis_function(sample_count):
    degree = min(3, sample_count - 1)
    knots = make_interp_spline(np.linspace(0.0, 1.0, sample_count), np.zeros(sample_count), k=degree).t
    position = ca.MX.sym('normalized_progress')
    # Fixed basis keeps scenario coefficients out of the differentiated spline node.
    basis = ca.bspline(position, ca.DM(np.eye(sample_count)).reshape((-1, 1)),
                       [knots.tolist()], [degree], sample_count, {})
    return ca.Function('progress_spline_basis', [position], [basis])


def progress_reference_values(progress, reference, sample_count, channels=None):
    normalized = (progress - reference[0]) / ca.fmax(reference[1] - reference[0], 1e-8)
    clamped = ca.fmin(1.0, ca.fmax(0.0, normalized))
    basis = spline_basis_function(sample_count).map(progress.numel())(ca.reshape(clamped, 1, -1))
    coefficients = ca.reshape(reference[2:], 6, sample_count)
    if channels is not None:
        coefficients = coefficients[list(channels), :]
    return (coefficients @ basis).T


def progress_reference_function(sample_count, channels=None):
    progress = ca.MX.sym('progress')
    reference = ca.MX.sym('reference', 2 + 6 * sample_count)
    values = progress_reference_values(progress, reference, sample_count, channels).T
    return ca.Function('progress_reference', [progress, reference], [values])


def fit_drivable_gp(waypoints, support_limit, kernel, cost, noise_std):
    supports = np.full((support_limit, 2), 1000.0)
    targets = np.zeros(support_limit)
    points = np.asarray(waypoints, dtype=float)
    if not points.size:
        return supports, targets
    points = np.unique(points[:, :2], axis=0)
    if not np.isfinite(points).all():
        raise ValueError('Drivable waypoints must be finite')
    if len(points) > support_limit:
        selected = [int(np.argmin(np.sum(points**2, axis=1)))]
        distance = np.sum((points - points[selected[0]])**2, axis=1)
        for _ in range(1, support_limit):
            selected.append(int(np.argmax(distance)))
            distance = np.minimum(distance, np.sum((points - points[selected[-1]])**2, axis=1))
        points = points[selected]
    supports[:len(points)] = points
    targets[:len(points)] = cost
    matrix = numeric_rbf_kernel(supports, supports, kernel)
    alpha = np.linalg.solve(matrix + noise_std**2 * np.eye(support_limit), targets)
    return supports, alpha