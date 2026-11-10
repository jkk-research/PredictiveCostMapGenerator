import hashlib
import ctypes
import os
from pathlib import Path
import sys

import casadi as ca
import numpy as np


class AcadosPolynomialPrototype:
    def __init__(self, problem, tolerance=0.001, root=None):
        root = Path(root or os.environ.get('ACADOS_SOURCE_DIR', Path.home() / '.cache/pcm_acados/acados'))
        if not (root / 'lib/libacados.so').exists():
            raise RuntimeError('acados is not installed; set ACADOS_SOURCE_DIR to a built checkout')
        os.environ['ACADOS_SOURCE_DIR'] = str(root)
        os.environ.setdefault('TERA_PATH', str(root / 't_renderer'))
        self.dependencies = [ctypes.CDLL(str(root / 'lib' / name), mode=ctypes.RTLD_GLOBAL)
                     for name in ('libblasfeo.so', 'libhpipm.so', 'libacados.so')]
        sys.path.insert(0, str(root / 'interfaces/acados_template'))
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

        self.values = ca.Function('acados_original_fg', [problem['x'], problem['p']], [problem['f'], problem['g']])
        objective_scale = 100.0
        signature = hashlib.sha256(self.values.serialize().encode() +
            f'full_dual_scaled_100_tolerance_{tolerance:.17g}'.encode()).hexdigest()[:16]
        directory = root.parent / 'models' / signature
        directory.mkdir(parents=True, exist_ok=True)
        model = AcadosModel()
        model.name = 'pcm_polynomial_' + signature
        model.x = ca.MX.sym('fixed_dummy_state', 1)
        model.u = problem['x']
        model.p = problem['p']
        model.disc_dyn_expr = ca.MX.zeros(1)
        model.cost_expr_ext_cost = objective_scale * problem['f']
        model.cost_expr_ext_cost_e = ca.MX(0)
        model.cost_expr_ext_cost_custom_hess = ca.DM.eye(model.x.numel() + model.u.numel())
        model.cost_expr_ext_cost_custom_hess_e = ca.DM.eye(model.x.numel())
        model.con_h_expr = problem['g']
        model.con_h_expr_0 = problem['g']
        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = 1
        ocp.solver_options.tf = 1.0
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.hessian_approx = 'EXACT'
        ocp.solver_options.exact_hess_dyn = 0
        ocp.solver_options.exact_hess_constr = 0
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        ocp.solver_options.globalization_full_step_dual = 1
        ocp.solver_options.nlp_solver_max_iter = 60
        ocp.solver_options.nlp_solver_tol_stat = objective_scale * tolerance
        ocp.solver_options.nlp_solver_tol_eq = tolerance
        ocp.solver_options.nlp_solver_tol_ineq = tolerance
        ocp.solver_options.nlp_solver_tol_comp = tolerance
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        count = problem['g'].numel()
        ocp.constraints.lh = np.full(count, -1e10)
        ocp.constraints.uh = np.zeros(count)
        ocp.constraints.lh_0 = np.full(count, -1e10)
        ocp.constraints.uh_0 = np.zeros(count)
        ocp.constraints.x0 = np.zeros(1)
        ocp.parameter_values = np.zeros(problem['p'].numel())
        ocp.code_export_directory = str(directory / 'generated')
        library = directory / 'generated' / ('libacados_ocp_solver_' + model.name + '.so')
        cached = library.exists() and (directory / 'ocp.json').exists()
        self.solver = AcadosOcpSolver(ocp, json_file=str(directory / 'ocp.json'), generate=not cached, build=not cached)
        self.tolerance = tolerance
        self.statistics = {}

    def stats(self):
        return self.statistics

    def __call__(self, x0, p, lbx=-np.inf, ubx=np.inf, lbg=-np.inf, ubg=0.0, **multipliers):
        if (np.any(np.isfinite(lbx)) or np.any(np.isfinite(ubx)) or
            np.any(np.isfinite(lbg)) or np.any(np.asarray(ubg) != 0)):
            raise ValueError('Prototype preserves the current unbounded-coefficient, g<=0 formulation only')
        for stage in (0, 1):
            self.solver.set(stage, 'p', np.asarray(p, dtype=float).reshape(-1))
            self.solver.set(stage, 'x', np.zeros(1))
        self.solver.set(0, 'u', np.asarray(x0, dtype=float).reshape(-1))
        status = self.solver.solve()
        coefficients = self.solver.get(0, 'u')
        objective, constraints = self.values(coefficients, p)
        finite = np.isfinite(coefficients).all() and np.isfinite(constraints.full()).all() and np.isfinite(float(objective))
        accepted = status == 0 and finite and np.max(constraints.full()) <= max(1e-6, 2 * self.tolerance)
        self.statistics = {'success': bool(accepted), 'return_status': 'Solve_Succeeded' if accepted else 'acados_status_' + str(status),
                           'iter_count': int(self.solver.get_stats('sqp_iter')),
                           't_wall_nlp_jac_fg': float(self.solver.get_stats('time_lin')),
                           'solver_seconds': float(self.solver.get_stats('time_tot')),
                           'residuals': self.solver.get_residuals().tolist()}
        return {'x': ca.DM(coefficients), 'f': objective, 'g': constraints,
                'lam_x': ca.DM.zeros(len(coefficients)), 'lam_g': ca.DM.zeros(constraints.numel())}