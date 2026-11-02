import casadi as ca
import numpy as np


def numeric_rbf_kernel(query, support, parameters):
    scaled = (np.asarray(query, dtype=float)[:, None, :] - np.asarray(support, dtype=float)[None, :, :]) \
        / np.asarray(parameters[1:1 + np.shape(query)[1]], dtype=float)
    return parameters[0]**2 * np.exp(-0.5 * np.sum(scaled**2, axis=2))


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