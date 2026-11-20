import casadi as ca
import numpy as np


def numeric_rbf_kernel(query, support, parameters):
    scaled = (np.asarray(query, dtype=float)[:, None, :] - np.asarray(support, dtype=float)[None, :, :]) \
        / np.asarray(parameters[1:1 + np.shape(query)[1]], dtype=float)
    return parameters[0]**2 * np.exp(-0.5 * np.sum(scaled**2, axis=2))


def compress_rbf_mean(points, alpha, parameters, groups, supports_per_path, matrix):
    selected = []
    for group in groups:
        candidates = points[group] / np.asarray(parameters[1:])
        chosen = [0]
        distance = np.sum((candidates - candidates[0])**2, axis=1)
        for _ in range(min(supports_per_path, len(group)) - 1):
            distance[chosen] = -1.0
            index = int(np.argmax(distance))
            chosen.append(index)
            distance = np.minimum(distance, np.sum((candidates - candidates[index])**2, axis=1))
        selected.extend(group[index] for index in sorted(chosen))
    selected = np.asarray(selected, dtype=int)
    if not len(selected):
        return selected, np.empty(0), 0.0
    if all(supports_per_path >= len(group) for group in groups):
        return selected, alpha[selected].copy(), 0.0
    reduced_kernel = matrix[np.ix_(selected, selected)]
    cross_mean = matrix[selected] @ alpha
    ridge = parameters[0]**2 * 1e-10
    beta = np.linalg.solve(reduced_kernel + ridge * np.eye(len(selected)), cross_mean)
    residual = alpha.copy()
    residual[selected] -= beta
    energy = float(residual @ matrix @ residual)
    roundoff = 64.0 * len(alpha) * np.finfo(float).eps * float(
        np.abs(residual) @ np.abs(matrix) @ np.abs(residual))
    bound = parameters[0] * np.sqrt(max(0.0, energy) + roundoff)
    if not np.isfinite(beta).all() or not np.isfinite(bound):
        raise ValueError('Sparse GP compression produced nonfinite coefficients or error bound')
    return selected, beta, float(bound)


def compress_global_rbf_mean(points, alpha, parameters, candidates, matrix,
                             minimum_supports=10, initial_fraction=0.2,
                             distance_threshold=0.75, error_tolerance=0.02):
    if (minimum_supports < 1 or not np.isfinite(initial_fraction) or not 0 <= initial_fraction <= 1
            or not np.isfinite(distance_threshold) or distance_threshold <= 0
            or not np.isfinite(error_tolerance) or error_tolerance < 0):
        raise ValueError('Invalid global GP reduction settings')
    candidates = np.unique(np.asarray(candidates, dtype=int))
    if not len(candidates):
        if np.any(alpha):
            raise ValueError('Global GP reduction needs candidates for a nonzero mean')
        return candidates, np.empty(0), 0.0
    scaled = points[candidates] / np.asarray(parameters[1:])
    source_mean = matrix[candidates] @ alpha
    chosen = [int(np.argmax(np.abs(source_mean)))]
    distance = np.sum((scaled - scaled[chosen[0]])**2, axis=1)
    seed_count = min(len(candidates), max(minimum_supports, int(np.ceil(initial_fraction * len(candidates)))))
    for _ in range(seed_count - 1):
        distance[chosen] = -1.0
        index = int(np.argmax(distance))
        if distance[index] <= 1e-24:
            break
        chosen.append(index)
        distance = np.minimum(distance, np.sum((scaled - scaled[index])**2, axis=1))
    while True:
        selected = candidates[np.asarray(chosen)]
        residual = alpha.copy()
        residual[selected] = 0.0
        if not np.any(residual):
            return selected, alpha[selected].copy(), 0.0
        reduced_kernel = matrix[np.ix_(selected, selected)]
        ridge = parameters[0]**2 * 1e-10
        beta = np.linalg.solve(reduced_kernel + ridge * np.eye(len(selected)), matrix[selected] @ alpha)
        residual = alpha.copy()
        residual[selected] -= beta
        energy = float(residual @ matrix @ residual)
        roundoff = 64.0 * len(alpha) * np.finfo(float).eps * float(
            np.abs(residual) @ np.abs(matrix) @ np.abs(residual))
        bound = parameters[0] * np.sqrt(max(0.0, energy) + roundoff)
        if not np.isfinite(beta).all() or not np.isfinite(bound):
            raise ValueError('Global GP compression produced nonfinite coefficients or error bound')
        distance[chosen] = -1.0
        if bound <= error_tolerance and np.max(distance) <= distance_threshold**2:
            return selected, beta, float(bound)
        remaining = np.flatnonzero(distance >= 0)
        if not len(remaining):
            return selected, beta, float(bound)
        separated = remaining[distance[remaining] > distance_threshold**2]
        eligible = separated if len(separated) else remaining
        errors = np.abs(matrix[candidates[eligible]] @ residual)
        index = int(eligible[np.argmax(errors)])
        chosen.append(index)
        distance = np.minimum(distance, np.sum((scaled - scaled[index])**2, axis=1))


def compress_hard_rbf_mean(points, alpha, parameters, candidates, max_supports):
    if max_supports < 1:
        raise ValueError('Hard GP support capacity must be positive')
    candidates = np.unique(np.asarray(candidates, dtype=int))
    scaled = points[candidates] / np.asarray(parameters[1:])
    chosen = []
    if len(candidates):
        chosen = [0]
        distance = np.sum((scaled - scaled[0])**2, axis=1)
        for _ in range(min(max_supports, len(candidates)) - 1):
            distance[chosen] = -1.0
            index = int(np.argmax(distance))
            if distance[index] <= 1e-24:
                break
            chosen.append(index)
            distance = np.minimum(distance, np.sum((scaled - scaled[index])**2, axis=1))
    selected = candidates[np.asarray(chosen, dtype=int)]
    return selected, alpha[selected].copy(), 0.0


def rbf_kernel(query, support, parameters):
    distance = 0
    for dimension in range(query.size2()):
        difference = (ca.repmat(query[:, dimension], 1, support.size1())
                       - ca.repmat(support[:, dimension].T, query.size1(), 1)) / parameters[dimension + 1]
        distance += difference**2
    return parameters[0]**2 * ca.exp(-0.5 * distance)


def lane_kernel(query, support, parameters):
    distance_x = (ca.repmat(query[:, 0], 1, support.size1())
                  - ca.repmat(support[:, 0].T, query.size1(), 1)) / parameters[1]
    distance_y = (ca.repmat(query[:, 1], 1, support.size1())
                  - ca.repmat(support[:, 1].T, query.size1(), 1)) / parameters[2]
    return parameters[0]**2 * ca.exp(-0.5 * (distance_x**2 + distance_y**2))