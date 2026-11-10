import casadi as ca
import numpy as np

from plan_predictive_cost_map.pcm_sqp import DenseSQPSolver


def solver(maximum_iterations=40):
    point = ca.MX.sym('point', 2)
    target = ca.MX.sym('target', 2)
    oracle = ca.nlpsol('sqp_test_oracle', 'sqpmethod', {
        'x': point, 'p': target, 'f': ca.sumsqr(point - target), 'g': ca.sum1(point)}, {
        'qpsol': 'qrqp', 'hessian_approximation': 'limited-memory',
        'print_header': False, 'print_iteration': False, 'print_status': False,
        'qpsol_options': {'print_header': False, 'print_iter': False}})
    return DenseSQPSolver(oracle, 1e-8, maximum_iterations)


def test_dense_sqp_uses_exact_constraints_and_shared_derivatives():
    optimize = solver()
    result = optimize(x0=[0.0, 0.0], p=[1.0, 1.0], ubg=1.0)
    assert optimize.stats()['success']
    np.testing.assert_allclose(result['x'].full().flatten(), [0.5, 0.5], atol=1e-7)
    assert float(result['g']) <= 1.0 + 1e-8
    assert optimize.stats()['n_call_nlp_jac_fg'] <= 4
    np.testing.assert_allclose(float(result['f']), 0.5, atol=1e-8)


def test_dense_sqp_does_not_cache_across_changed_parameters():
    optimize = solver()
    optimize(x0=[0.0, 0.0], p=[1.0, 1.0], ubg=1.0)
    result = optimize(x0=[0.5, 0.5], p=[-2.0, -3.0], lbx=[-1.0, -4.0],
                      ubx=[4.0, 4.0], lbg=-4.0, ubg=2.0)
    assert optimize.stats()['success']
    np.testing.assert_allclose(result['x'].full().flatten(), [-1.0, -3.0], atol=1e-7)


def test_dense_sqp_reports_iteration_limit_as_failure():
    optimize = solver(maximum_iterations=1)
    optimize(x0=[10.0, 10.0], p=[1.0, 1.0], ubg=1.0)
    assert not optimize.stats()['success']
    assert optimize.stats()['return_status'] != 'Solve_Succeeded'