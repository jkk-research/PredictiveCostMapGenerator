import os

import casadi as ca
import numpy as np
import pytest

from plan_predictive_cost_map.pcm_acados import AcadosPolynomialPrototype


@pytest.mark.skipif(os.environ.get('PCM_ACADOS_TEST') != '1', reason='Opt-in acados code generation')
def test_acados_prototype_preserves_original_constraints():
    point = ca.MX.sym('coefficients', 2)
    target = ca.MX.sym('parameters', 2)
    solver = AcadosPolynomialPrototype({'x': point, 'p': target,
        'f': ca.sumsqr(point - target), 'g': ca.sum1(point) - 1.0}, tolerance=1e-6)
    result = solver(x0=[0.0, 0.0], p=[1.0, 1.0])
    assert solver.stats()['success'], solver.stats()
    np.testing.assert_allclose(result['x'].full().reshape(-1), [0.5, 0.5], atol=1e-4)
    np.testing.assert_allclose(float(result['f']), 0.5, atol=1e-5)
    assert float(result['g']) <= 2e-6


@pytest.mark.skipif(os.environ.get('PCM_ACADOS_PLANNER_TEST') != '1', reason='Opt-in short-horizon planner prototype')
def test_acados_preserves_actuator_planner_problem():
    from pathlib import Path
    import runpy
    from timeit import default_timer
    import rclpy
    from crp_msgs.msg import ActuatorState
    from plan_predictive_cost_map.pcm_longitudinal import pcm_node

    rclpy.init(args=['--ros-args', '-p', 'prediction_model:=dynamic_actuators',
        '-p', 'lookahead_time:=0.5', '-p', 'prediction_steps:=5', '-p', 'actuator_solver.method:=sqp',
        '-p', 'actuator_solver.jit:=false', '-p', 'scenario_worker:=false', '-p', 'sparse_gp:=false'])
    node = pcm_node()
    try:
        fixture = runpy.run_path(str(Path(__file__).with_name('benchmark_pcm_sparse.py')))
        scenario, _, _ = fixture['scenario_inputs']('clear', node.tau_h)
        node.scenario_callback(scenario)
        node.ego_state.vx = 8.0
        node.powertrain_callback(ActuatorState(motor_speed_rad_s=225.0))
        node.last_exec = 0.0
        node.callback()
        assert node.plan_diagnostics['accepted']
        objective, constraints = node.solver.get_function('nlp_fg')(node.c_sym, node.p_sym)
        solver = AcadosPolynomialPrototype({'x': node.c_sym, 'p': node.p_sym,
                                            'f': objective, 'g': constraints}, tolerance=node.solver_tolerance)
        started = default_timer()
        result = solver(x0=node.last_c_opt, p=node.p_numeric)
        elapsed = default_timer() - started
        assert solver.stats()['success'], solver.stats()
        assert np.max(result['g'].full()) <= 2 * node.solver_tolerance
        print('Short-horizon acados:', {'solve_ms': elapsed * 1000,
              'iterations': solver.stats()['iter_count'], 'maximum_constraint': float(np.max(result['g'].full())),
              'residuals': solver.stats()['residuals']})
    finally:
        node.destroy_node()
        rclpy.shutdown()