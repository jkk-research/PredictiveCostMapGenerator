import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import casadi as ca
import numpy as np

from plan_predictive_cost_map.pcm_lane_geometry import progress_reference_function
from plan_predictive_cost_map.pcm_kernels import lane_kernel, rbf_kernel
from plan_predictive_cost_map.pcm_powertrain import build_actuator_interval, motor_efficiency_function


def build_stage(node):
    state = ca.MX.sym('stage_state', 15)
    command = ca.MX.sym('stage_speed_command', 1)
    parameters = ca.MX.sym('stage_parameters', node.p_sym.numel())
    count = node.pedestrian_support_count
    vehicle_count = node.vehicle_support_count
    geometry_offset = 13 + 4 * count + 4 * vehicle_count
    reference_offset = geometry_offset + 3 * node.target_support_points + 3 * node.drivable_support_points
    reference = parameters[reference_offset:reference_offset + 2 + 6 * node.reference_samples]
    context_offset = reference_offset + reference.numel()
    context = parameters[context_offset:context_offset + 17]
    pitch_initial = context[2]
    cosine_pitch, sine_pitch = ca.cos(pitch_initial), ca.sin(pitch_initial)
    lookup = progress_reference_function(node.reference_samples)

    def position(joint):
        return ca.vertcat(cosine_pitch * joint[0] + sine_pitch * joint[2], joint[1], lookup(joint[13], reference)[2])

    def guidance(joint):
        geometry = lookup(joint[13], reference)
        tangent_x = ca.cos(geometry[4]) * ca.cos(geometry[3])
        tangent_y = ca.cos(geometry[4]) * ca.sin(geometry[3])
        tangent_z = ca.sin(geometry[4])
        vertical = sine_pitch * tangent_x + cosine_pitch * tangent_z
        horizontal = ca.sqrt(ca.fmax((cosine_pitch * tangent_x - sine_pitch * tangent_z)**2 + tangent_y**2, 1e-12))
        pitch = ca.atan2(vertical, horizontal)
        lookahead = ca.fmax(node.vehicle_settings.minimum_lookahead, node.vehicle_settings.lookahead_time * joint[4])
        target = lookup(joint[13] + lookahead, reference)
        target_x = cosine_pitch * target[0] - sine_pitch * target[2]
        target_z = sine_pitch * target[0] + cosine_pitch * target[2]
        delta_x, delta_y = target_x - joint[0], target[1] - joint[1]
        forward = ca.cos(pitch) * (ca.cos(joint[3]) * delta_x + ca.sin(joint[3]) * delta_y) + ca.sin(pitch) * (target_z - joint[2])
        lateral = -ca.sin(joint[3]) * delta_x + ca.cos(joint[3]) * delta_y
        distance = forward**2 + lateral**2
        steering = ca.if_else(distance < 1e-8, 0.0,
            ca.atan2(2.0 * node.vehicle_settings.wheelbase * lateral, ca.fmax(distance, 1e-8)))
        steering = ca.fmin(node.vehicle_settings.maximum_steering, ca.fmax(-node.vehicle_settings.maximum_steering, steering))
        return ca.vertcat(pitch, pitch if node.use_road_gradient else node.model_road_pitch), steering

    efficiency = motor_efficiency_function(node.motor_efficiency_map, node.efficiency_model,
                                          node.efficiency_degree, node.efficiency_tolerance)
    duration = node.tau_h / node.N
    ticks = max(1, math.ceil(duration / node.model_control_step))
    interval = build_actuator_interval(duration / ticks, node.tau_h, 0.0, efficiency,
        node.powertrain_settings, node.vehicle_settings, node.actuator_vehicle_model,
        node.actuator_integration, node.actuator_stage_step, held_speed_command=True)
    tick_state = ca.MX.sym('tick_state', 15)
    tick_command = ca.MX.sym('tick_command')
    road, steering = guidance(tick_state)
    next_tick, tick_acceleration, _ = interval(tick_state, tick_command, parameters[0], road, steering)
    tick = ca.Function('actuator_speed_tick', [tick_state, tick_command, parameters],
                       [next_tick, tick_acceleration, steering])
    history, acceleration, steering_history = tick.mapaccum(ticks)(state,
        ca.repmat(command, 1, ticks), ca.repmat(parameters, 1, ticks))
    final = history[:, -1]
    query_position = position(final)
    scene_yaw = context[15]
    scene_x = context[13] + ca.cos(scene_yaw) * query_position[0] - ca.sin(scene_yaw) * query_position[1]
    scene_y = context[14] + ca.sin(scene_yaw) * query_position[0] + ca.cos(scene_yaw) * query_position[1]
    query = ca.horzcat(scene_x, scene_y, final[14] + context[16])
    vehicle_offset = 13 + 4 * count
    pedestrian_points = ca.reshape(parameters[13 + count:13 + 4 * count], count, 3)
    vehicle_points = ca.reshape(parameters[vehicle_offset + vehicle_count:geometry_offset], vehicle_count, 3)
    pedestrian = rbf_kernel(query, pedestrian_points, node.p_ped) @ parameters[13:13 + count] - 1.0
    vehicle = rbf_kernel(query, vehicle_points, node.p_veh) @ parameters[vehicle_offset:vehicle_offset + vehicle_count] - 1.0
    if node.sparse_gp:
        pedestrian += parameters[-3]
        vehicle += parameters[-2]
    target_end = geometry_offset + 2 * node.target_support_points
    target_alpha_end = target_end + node.target_support_points
    road_end = target_alpha_end + 2 * node.drivable_support_points
    target = lane_kernel(query[:, :2], ca.reshape(parameters[geometry_offset:target_end], node.target_support_points, 2), node.p_el) @ parameters[target_end:target_alpha_end]
    drivable = lane_kernel(query[:, :2], ca.reshape(parameters[target_alpha_end:road_end], node.drivable_support_points, 2), node.p_ol) @ parameters[road_end:reference_offset]
    field = node.smooth_max(node.smooth_max(pedestrian, vehicle), ca.fmin(target, drivable))
    physical_speed = history[4, :].T
    limits = ca.vertcat(acceleration[0, :].T - node.ax_max, -node.ax_max - acceleration[0, :].T,
        acceleration[1, :].T - node.ay_max, -node.ay_max - acceleration[1, :].T,
        -physical_speed, physical_speed - ca.fmax(parameters[0], parameters[-1]),
        field - node.max_gp_cost, final[13] - reference[1])
    residuals = ca.vertcat(
        math.sqrt(node.get_parameter('weight_safety').value / node.N) * (field + 1.0),
        math.sqrt(node.get_parameter('weight_velocity').value / node.N) *
            (parameters[-1] - final[4]) / ca.sqrt(ca.fmax(parameters[-1]**2, 1.0)),
        math.sqrt(node.get_parameter('weight_lateral_acceleration').value / node.N) * acceleration[1, -1] / node.ay_max,
        math.sqrt(node.get_parameter('weight_acceleration').value / (node.N * node.ax_max)) * acceleration[0, -1])
    crash = node.get_parameter('weight_crash').value * (ca.exp(5 * node.softplus(field)) - 1.0)
    net = (final[11] - final[12]) - (state[11] - state[12])
    cost = ca.sumsqr(residuals) + crash + node.weight_energy * net / node.energy_reference_j
    pitch = guidance(final)[0][0]
    quaternion = ca.vertcat(ca.sin(final[3] / 2) * ca.sin(pitch / 2),
        -ca.cos(final[3] / 2) * ca.sin(pitch / 2), ca.sin(final[3] / 2) * ca.cos(pitch / 2),
        ca.cos(final[3] / 2) * ca.cos(pitch / 2))
    initial_quaternion = ca.vertcat(0.0, -ca.sin(pitch_initial / 2), 0.0, ca.cos(pitch_initial / 2))
    relative_vector = initial_quaternion[3] * quaternion[:3] - quaternion[3] * initial_quaternion[:3] - ca.cross(initial_quaternion[:3], quaternion[:3])
    relative_scalar = initial_quaternion[3] * quaternion[3] + ca.dot(initial_quaternion[:3], quaternion[:3])
    report = ca.Function('acados_velocity_stage', [state, command, parameters],
        [final, acceleration[:, -1], field, steering_history[0, -1], query_position,
         ca.vertcat(relative_vector, relative_scalar), limits, cost])
    return state, command, parameters, final, limits, cost, residuals, report


def export_stages(node, directory):
    root = Path(os.environ.get('ACADOS_SOURCE_DIR', Path.home() / '.cache/pcm_acados/acados'))
    os.environ['ACADOS_SOURCE_DIR'] = str(root)
    os.environ.setdefault('TERA_PATH', str(root / 't_renderer'))
    dependencies = [ctypes.CDLL(str(root / 'lib' / name), mode=ctypes.RTLD_GLOBAL)
                    for name in ('libblasfeo.so', 'libhpipm.so', 'libacados.so')]
    sys.path.insert(0, str(root / 'interfaces/acados_template'))
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

    state, command, parameters, dynamics, constraints, cost, residuals, report = build_stage(node)
    signature = hashlib.sha256(report.serialize().encode() +
        f'{node.N}_{node.tau_h}_{node.solver_tolerance}_stages_v1'.encode()).hexdigest()[:16]
    name = 'pcm_velocity_' + signature
    generated = Path(directory) / 'acados_generated'
    model = AcadosModel()
    model.name = name
    model.x, model.u, model.p = state, command, parameters
    model.disc_dyn_expr = dynamics
    model.cost_expr_ext_cost = 100.0 * cost
    model.cost_expr_ext_cost_e = ca.MX(0)
    variables = ca.vertcat(command, state)
    jacobian = ca.jacobian(residuals, variables)
    model.cost_expr_ext_cost_custom_hess = 200.0 * jacobian.T @ jacobian + 1e-4 * ca.DM.eye(16)
    model.cost_expr_ext_cost_custom_hess_e = 1e-4 * ca.DM.eye(15)
    model.con_h_expr = constraints
    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = node.N
    ocp.solver_options.tf = node.tau_h
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_cond_N = min(node.N, 5)
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.exact_hess_dyn = 0
    ocp.solver_options.exact_hess_constr = 0
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.globalization_full_step_dual = 1
    ocp.solver_options.nlp_solver_max_iter = 40
    ocp.solver_options.nlp_solver_tol_stat = 100 * node.solver_tolerance
    ocp.solver_options.nlp_solver_tol_eq = node.solver_tolerance
    ocp.solver_options.nlp_solver_tol_ineq = node.solver_tolerance
    ocp.solver_options.nlp_solver_tol_comp = node.solver_tolerance
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.solver_options.cost_scaling = np.r_[np.ones(node.N), 1.0]
    ocp.constraints.x0 = np.zeros(15)
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbu = np.zeros(1)
    ocp.constraints.ubu = np.array([100.0])
    ocp.constraints.lh = np.full(constraints.numel(), -1e10)
    ocp.constraints.uh = np.zeros(constraints.numel())
    ocp.parameter_values = np.zeros(parameters.numel())
    ocp.code_export_directory = str(generated)
    library = generated / ('libacados_ocp_solver_' + name + '.so')
    cached = library.exists() and (Path(directory) / 'acados_ocp.json').exists()
    solver = AcadosOcpSolver(ocp, json_file=str(Path(directory) / 'acados_ocp.json'), generate=not cached, build=not cached)
    report.save(str(Path(directory) / 'stage.casadi'))
    template = Path(__file__).resolve().parents[1] / 'resource/acados_stage_runtime.c.in'
    if not template.exists():
        from ament_index_python.packages import get_package_share_directory
        template = Path(get_package_share_directory('plan_predictive_cost_map')) / 'resource/acados_stage_runtime.c.in'
    runtime_source = Path(directory) / 'stage_runtime.c'
    runtime_source.write_text(template.read_text().replace('@MODEL@', name).replace('@STAGES@', str(node.N))
                              .replace('@PARAMETERS@', str(parameters.numel())))
    runtime_library = Path(directory) / 'stage_runtime.so'
    subprocess.run(['gcc', '-O2', '-fPIC', '-shared', str(runtime_source), '-I' + str(generated),
        '-I' + str(root / 'include'), '-I' + str(root / 'include/blasfeo/include'),
        '-I' + str(root / 'include/hpipm/include'), '-L' + str(generated),
        '-lacados_ocp_solver_' + name, '-L' + str(root / 'lib'), '-lacados', '-lm',
        '-Wl,-rpath,' + str(generated), '-Wl,-rpath,' + str(root / 'lib'), '-o', str(runtime_library)], check=True)
    return {'solver': solver, 'name': name, 'library': str(library), 'generated': str(generated),
            'acados_lib_dir': str(root / 'lib'), 'stage': report, 'dependencies': dependencies,
            'runtime_library': str(runtime_library)}