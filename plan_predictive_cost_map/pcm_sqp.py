from timeit import default_timer

import casadi as ca
import numpy as np
from scipy.optimize import Bounds, minimize


class DenseSQPSolver:
    def __init__(self, oracle, tolerance, maximum_iterations=40):
        self.oracle = oracle
        self.tolerance = tolerance
        self.maximum_iterations = maximum_iterations
        self.statistics = {}
        self.compatibility_functions = {}

    def get_function(self, name):
        if name in ('nlp_f', 'nlp_g', 'nlp_grad_f', 'nlp_jac_g'):
            if name not in self.compatibility_functions:
                function = self.oracle.get_function('nlp_jac_fg')
                inputs = [ca.MX.sym(function.name_in(index), function.sparsity_in(index)) for index in range(function.n_in())]
                outputs = function(*inputs)
                selected = {'nlp_f': [outputs[0]], 'nlp_g': [outputs[2]],
                            'nlp_grad_f': outputs[:2], 'nlp_jac_g': outputs[2:]}[name]
                output_names = {'nlp_f': ['f'], 'nlp_g': ['g'], 'nlp_grad_f': ['f', 'grad_f'],
                                'nlp_jac_g': ['g', 'jac_g']}[name]
                self.compatibility_functions[name] = ca.Function(name, inputs, selected,
                    [function.name_in(index) for index in range(function.n_in())], output_names)
            return self.compatibility_functions[name]
        return self.oracle.get_function(name)

    def stats(self):
        return self.statistics

    def __call__(self, x0, p, lbx=-np.inf, ubx=np.inf, lbg=-np.inf, ubg=0.0, **multipliers):
        value_function = self.get_function('nlp_fg')
        derivative_function = self.get_function('nlp_jac_fg')
        initial = np.asarray(x0, dtype=float).reshape(-1)
        stats = {'t_wall_nlp_fg': 0.0, 't_wall_nlp_jac_fg': 0.0,
                 'n_call_nlp_fg': 0, 'n_call_nlp_jac_fg': 0}
        cached_point, cached_values, cached_derivatives = None, None, None

        def evaluate(point, derivatives=False):
            nonlocal cached_point, cached_values, cached_derivatives
            if cached_point is None or not np.array_equal(point, cached_point):
                cached_point, cached_values, cached_derivatives = point.copy(), None, None
            if derivatives and cached_derivatives is None:
                started = default_timer()
                objective, gradient, constraints, jacobian = derivative_function(point, p)
                stats['t_wall_nlp_jac_fg'] += default_timer() - started
                stats['n_call_nlp_jac_fg'] += 1
                cached_values = float(objective), constraints.full().reshape(-1)
                cached_derivatives = gradient.full().reshape(-1), jacobian.full()
            elif cached_values is None:
                started = default_timer()
                objective, constraints = value_function(point, p)
                stats['t_wall_nlp_fg'] += default_timer() - started
                stats['n_call_nlp_fg'] += 1
                cached_values = float(objective), constraints.full().reshape(-1)
            return cached_derivatives if derivatives else cached_values

        initial_constraints = evaluate(initial)[1]
        lower = np.broadcast_to(np.asarray(lbg, dtype=float), initial_constraints.shape)
        upper = np.broadcast_to(np.asarray(ubg, dtype=float), initial_constraints.shape)
        lower_finite, upper_finite = np.isfinite(lower), np.isfinite(upper)

        def residuals(point):
            values = evaluate(point)[1]
            return np.r_[values[lower_finite] - lower[lower_finite],
                         upper[upper_finite] - values[upper_finite]]

        def jacobian(point):
            values = evaluate(point, True)[1]
            return np.vstack((values[lower_finite], -values[upper_finite]))

        result = minimize(lambda point: evaluate(point)[0], initial,
            jac=lambda point: evaluate(point, True)[0], method='SLSQP',
            bounds=Bounds(np.broadcast_to(lbx, initial.shape), np.broadcast_to(ubx, initial.shape)),
            constraints={'type': 'ineq', 'fun': residuals, 'jac': jacobian},
            options={'ftol': self.tolerance, 'maxiter': self.maximum_iterations})
        objective, constraints = evaluate(result.x)
        self.statistics = {**stats, 'success': bool(result.success), 'iter_count': int(result.nit),
                           'return_status': 'Solve_Succeeded' if result.success else str(result.message)}
        return {'x': ca.DM(result.x), 'f': ca.DM(objective), 'g': ca.DM(constraints),
                'lam_x': ca.DM.zeros(len(initial)), 'lam_g': ca.DM.zeros(len(constraints))}