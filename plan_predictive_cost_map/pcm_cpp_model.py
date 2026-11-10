import argparse
import json
from pathlib import Path

import casadi as ca
import rclpy

from plan_predictive_cost_map.pcm_lane_geometry import spline_basis_function
from plan_predictive_cost_map.pcm_longitudinal import pcm_node


def export_model(parameters, directory, representation='polynomial'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arguments = ['--ros-args']
    settings = {**parameters, 'scenario_worker': False, 'publish_gp_state': False}
    if representation == 'acados_stages':
        settings['actuator_solver.jit'] = False
    for name, value in settings.items():
        arguments.extend(['-p', name + ':=' + (json.dumps(value) if value != '' else "''")])
    rclpy.init(args=arguments)
    node = pcm_node()
    try:
        solver = getattr(node.solver, 'oracle', node.solver)
        solver.save(str(directory / 'solver.casadi'))
        warm = getattr(node, 'light_warm_solver', solver)
        getattr(warm, 'oracle', warm).save(str(directory / 'warm.casadi'))
        if node.uses_sqp:
            node.actuator_retry_solver.save(str(directory / 'retry.casadi'))
            objective_constraints = solver.get_function('nlp_fg')
        else:
            objective_constraints = ca.Function('native_fg', [node.c_sym, node.p_sym], [
                solver.get_function('nlp_f')(node.c_sym, node.p_sym),
                solver.get_function('nlp_g')(node.c_sym, node.p_sym)])
        objective_constraints.save(str(directory / 'fg.casadi'))
        if node.uses_sqp:
            derivatives = solver.get_function('nlp_jac_fg')
        else:
            objective, constraints = objective_constraints(node.c_sym, node.p_sym)
            derivatives = ca.Function('native_derivatives', [node.c_sym, node.p_sym],
                [objective, ca.gradient(objective, node.c_sym), constraints, ca.jacobian(constraints, node.c_sym)])
        derivatives.save(str(directory / 'jac.casadi'))
        prediction = node.rollout_eval(node.c_sym, node.p_sym)
        if node.actuators_enabled:
            node.reporting_eval.save(str(directory / 'report.casadi'))
        else:
            reporting = ca.Function('native_reporting', [node.c_sym, node.p_sym],
                                    [*prediction, *node.fsup_eval(node.c_sym, node.p_sym)])
            reporting.save(str(directory / 'report.casadi'))
        spline_basis_function(node.reference_samples).save(str(directory / 'basis.casadi'))
        values = {name: parameter.value for name, parameter in node._parameters.items()}
        values.update(parameters)
        manifest = {'casadi_version': ca.__version__, 'parameters': values,
                    'decision_count': node.M, 'parameter_count': node.p_sym.numel(),
                    'constraint_count': node.g.numel(), 'prediction_output_count': len(prediction),
                    'pedestrian_support_count': node.pedestrian_support_count,
                    'vehicle_support_count': node.vehicle_support_count,
                    'actuators_enabled': node.actuators_enabled,
                    'solver_method': 'sqp' if node.uses_sqp else 'ipopt',
                    'exact_gp_capacity': node.exact_gp_capacity}
        manifest['velocity_representation'] = representation
        if representation == 'acados_stages':
            if not node.actuators_enabled:
                raise ValueError('acados_stages currently requires prediction_model=dynamic_actuators')
            from plan_predictive_cost_map.pcm_acados_stages import export_stages
            stages = export_stages(node, directory)
            manifest.update(decision_count=node.N, solver_method='acados_stages',
                            acados_runtime_library=stages['runtime_library'], acados_lib_dir=stages['acados_lib_dir'])
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return str(directory.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    export_model(json.loads(arguments.parameters.read_text()), arguments.output)


if __name__ == '__main__':
    main()