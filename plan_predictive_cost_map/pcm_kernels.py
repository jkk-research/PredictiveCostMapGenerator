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