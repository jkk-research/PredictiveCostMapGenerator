from pathlib import Path
import subprocess

import numpy as np

from plan_predictive_cost_map.pcm_kernels import compress_rbf_mean, numeric_rbf_kernel


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