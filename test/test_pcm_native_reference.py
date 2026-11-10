import json
import os
from pathlib import Path
import subprocess

import numpy as np

from plan_predictive_cost_map.pcm_lane_geometry import fit_drivable_gp, prepare_progress_reference, spline_basis_function


def test_native_reference_and_static_gp_match_python(tmp_path):
    root = Path(__file__).resolve().parents[3]
    executable = root / 'install/plan_predictive_cost_map/lib/plan_predictive_cost_map/pcm_longitudinal_cpp'
    spline_basis_function(96).save(str(tmp_path / 'basis.casadi'))
    parameter = np.linspace(-0.5, 2.0, 180)
    route = np.column_stack((20 * np.sin(parameter), 20 * (1 - np.cos(parameter)), 0.5 * parameter**2))
    request = {'route': route.tolist(), 'samples': 96, 'supports': 32,
               'kernel': [0.4, 2.7, 1.0], 'cost': -1.0, 'noise': 0.001}
    result = subprocess.run([str(executable), '--preprocess-probe', str(tmp_path)],
                            input=json.dumps(request), text=True, capture_output=True, check=True)
    actual = json.loads(result.stdout)
    points, alpha = fit_drivable_gp(route[:, :2], 32, request['kernel'], -1.0, 0.001)
    np.testing.assert_allclose(actual['reference'], prepare_progress_reference(route, 96), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(actual['points'], points, atol=1e-12)
    np.testing.assert_allclose(actual['alpha'], alpha, rtol=1e-7, atol=1e-7)