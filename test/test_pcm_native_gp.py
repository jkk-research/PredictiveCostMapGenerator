from pathlib import Path
import subprocess

import numpy as np
import pytest

from plan_predictive_cost_map.pcm_kernels import compress_global_rbf_mean, compress_rbf_mean, numeric_rbf_kernel


def test_native_gp_fit_and_projection_match_python(tmp_path):
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / 'gp_native_probe'
    subprocess.run(['g++', '-std=c++17', '-O2', '-I/usr/include/eigen3', '-I' + str(root / 'include'),
                    str(root / 'test/gp_native_probe.cpp'), '-o', str(executable)], check=True)
    actual = np.fromstring(subprocess.check_output([str(executable)], text=True), sep=' ')
    indices = np.arange(10)
    points = np.column_stack((3.0 * indices, -2.0 + 0.2 * indices, 0.5 * indices))
    parameters = [0.8, 2.7, 2.2, 1.0]
    matrix = numeric_rbf_kernel(points, points, parameters)
    alpha = np.linalg.solve(matrix + np.diag(np.linspace(0.05, 0.2, 10)), np.linspace(1.0, 1.8, 10))
    selected, beta, bound = compress_rbf_mean(points, alpha, parameters, [indices], 5, matrix)
    expected = np.r_[selected, alpha, beta, bound]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_global_projection_shares_supports_across_overlapping_paths():
    times = np.linspace(0.2, 2.0, 20)
    path = np.column_stack((np.full(20, 12.0), -2.0 + times, times))
    points = np.vstack((path, path, path))
    parameters = [1.0, 3.5, 1.5, 2.0]
    matrix = numeric_rbf_kernel(points, points, parameters)
    alpha = np.linalg.solve(matrix + 0.05 * np.eye(len(points)), np.full(len(points), 3.0))
    selected, beta, bound = compress_global_rbf_mean(
        points, alpha, parameters, np.arange(len(points)), matrix,
        minimum_supports=2, initial_fraction=0, distance_threshold=0.75, error_tolerance=0.02)
    assert 2 <= len(selected) < len(path)
    assert bound <= 0.02
    queries = np.random.default_rng(42).uniform([0, -5, 0], [25, 5, 3], (1000, 3))
    original = numeric_rbf_kernel(queries, points, parameters) @ alpha
    reduced = numeric_rbf_kernel(queries, points[selected], parameters) @ beta
    assert np.max(np.abs(original - reduced)) <= bound + 1e-10
    assert np.min(reduced + bound - original) >= -1e-10
    selected_again, beta_again, bound_again = compress_global_rbf_mean(
        points, alpha, parameters, np.arange(len(points)), matrix,
        minimum_supports=2, initial_fraction=0, distance_threshold=0.75, error_tolerance=0.02)
    np.testing.assert_array_equal(selected, selected_again)
    np.testing.assert_array_equal(beta, beta_again)
    assert bound == bound_again


def test_global_projection_adds_nearby_supports_when_mean_error_requires_them():
    points = np.column_stack((np.linspace(0, 1, 10), np.zeros((10, 2))))
    parameters = [1.0, 1.0, 1.0, 1.0]
    matrix = numeric_rbf_kernel(points, points, parameters)
    alpha = np.arange(1.0, 11.0) * np.where(np.arange(10) % 2, -1, 1)
    loose = compress_global_rbf_mean(points, alpha, parameters, np.arange(10), matrix,
        minimum_supports=1, initial_fraction=0, distance_threshold=100, error_tolerance=10)
    strict = compress_global_rbf_mean(points, alpha, parameters, np.arange(10), matrix,
        minimum_supports=1, initial_fraction=0, distance_threshold=100, error_tolerance=0)
    assert len(loose[0]) < len(strict[0])
    assert strict[2] == 0
    np.testing.assert_allclose(matrix[:, strict[0]] @ strict[1], matrix @ alpha, atol=1e-12)


def test_global_projection_distance_coverage_is_kernel_scaled_including_time():
    points = np.array([[0, 0, 0], [0, 0, 0.1], [0, 0, 5.0], [0, 0, 5.1]])
    parameters = [1.0, 100.0, 100.0, 0.5]
    matrix = numeric_rbf_kernel(points, points, parameters)
    selected, _, _ = compress_global_rbf_mean(points, np.ones(4), parameters, np.arange(4), matrix,
        minimum_supports=1, initial_fraction=0, distance_threshold=0.5, error_tolerance=10)
    scaled = points / parameters[1:]
    nearest = np.min(np.linalg.norm(scaled[:, None] - scaled[selected][None], axis=2), axis=1)
    assert nearest.max() <= 0.5
    assert len(selected) >= 2


@pytest.mark.parametrize('settings', [dict(minimum_supports=0), dict(initial_fraction=1.1),
    dict(initial_fraction=float('nan')), dict(distance_threshold=0), dict(error_tolerance=-1)])
def test_global_projection_rejects_invalid_settings(settings):
    points = np.zeros((2, 3))
    with pytest.raises(ValueError):
        compress_global_rbf_mean(points, np.ones(2), [1, 1, 1, 1], np.arange(2), np.ones((2, 2)), **settings)


def test_global_projection_empty_and_full_capacity_are_exact():
    points = np.column_stack((np.arange(5), np.zeros((5, 2))))
    parameters = [1.0, 1.0, 1.0, 1.0]
    matrix = numeric_rbf_kernel(points, points, parameters)
    selected, beta, bound = compress_global_rbf_mean(points, np.ones(5), parameters, np.arange(5), matrix)
    assert bound == 0
    np.testing.assert_allclose(matrix[:, selected] @ beta, matrix @ np.ones(5))
    selected, beta, bound = compress_global_rbf_mean(points, np.zeros(5), parameters, [], matrix)
    assert len(selected) == len(beta) == 0 and bound == 0