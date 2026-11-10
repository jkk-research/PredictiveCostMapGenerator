from pathlib import Path
import subprocess

import casadi as ca
import numpy as np


def test_native_slsqp_matches_constrained_reference(tmp_path):
    root = Path(__file__).resolve().parents[1]
    workspace = root.parents[1]
    point = ca.MX.sym('coefficients', 2)
    parameters = ca.MX.sym('parameters', 2)
    objective = ca.sumsqr(point - parameters)
    constraints = ca.vertcat(ca.sum1(point) - 1.0)
    ca.Function('fg', [point, parameters], [objective, constraints]).save(str(tmp_path / 'fg.casadi'))
    ca.Function('native_derivatives', [point, parameters], [objective, ca.gradient(objective, point), constraints,
                ca.jacobian(constraints, point)]).save(str(tmp_path / 'jac.casadi'))
    ca.nlpsol('native_ipopt_reference', 'ipopt', {'x': point, 'p': parameters, 'f': objective, 'g': constraints},
              {'ipopt.print_level': 0, 'print_time': False, 'ipopt.sb': 'yes'}).save(str(tmp_path / 'ipopt.casadi'))
    native = Path.home() / '.cache/pcm_native/install'
    library = workspace / 'install/plan_predictive_cost_map/lib'
    executable = tmp_path / 'native_solver_probe'
    subprocess.run(['g++', '-std=c++17', '-O2', '-I/usr/include/eigen3', '-I' + str(root / 'include'),
        '-I' + str(native / 'include'), str(root / 'test/native_solver_probe.cpp'), '-L' + str(library),
        '-L' + str(native / 'lib'), '-lpcm_casadi_bridge', '-lnlopt',
        '-Wl,-rpath,' + str(library), '-Wl,-rpath,' + str(native / 'lib'), '-o', str(executable)], check=True)
    output = subprocess.check_output([str(executable), str(tmp_path)], text=True)
    np.testing.assert_allclose(np.fromstring(output, sep=' '), [0.5, 0.5, 0.5], atol=1e-7)