from pathlib import Path

import casadi as ca
import numpy as np
import os
import pytest

from plan_predictive_cost_map.pcm_powertrain import build_actuator_interval, motor_efficiency_function


def test_held_speed_stage_matches_constant_polynomial_command():
    root = Path(__file__).resolve().parents[2]
    efficiency = motor_efficiency_function(root / 'CooperativeResearchPlatform_Simulator/simulator/config/motor_efficiency.csv')
    held = build_actuator_interval(0.05, 1.0, 0.25, efficiency, method='discrete', held_speed_command=True)
    original = build_actuator_interval(0.05, 1.0, 0.25, efficiency, method='discrete')
    initial = np.zeros(15)
    initial[4] = 8.0
    initial[7] = 8.0 * 9.0 / 0.32
    actual = held(initial, [8.0], 8.0, [0.0, 0.0], 0.0)
    expected = original(initial, [0.0, 0.0, 0.0, 0.0], 8.0, [0.0, 0.0], 0.0)
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result.full(), reference.full(), rtol=1e-12, atol=1e-12)
    braking = held(initial, [7.0], 8.0, [0.0, 0.0], 0.0)
    assert float(braking[0][4]) < float(actual[0][4])
    assert held.size1_in(1) == 1


def test_stage_model_matches_constant_command_rollout():
    import runpy
    import rclpy
    from crp_msgs.msg import ActuatorState
    from plan_predictive_cost_map.pcm_longitudinal import pcm_node
    from plan_predictive_cost_map.pcm_acados_stages import build_stage

    rclpy.init(args=['--ros-args', '-p', 'prediction_model:=dynamic_actuators',
        '-p', 'lookahead_time:=0.2', '-p', 'command_lookahead_time:=0.1', '-p', 'prediction_steps:=2', '-p', 'actuator_solver.jit:=false',
        '-p', 'actuator_solver.method:=sqp', '-p', 'scenario_worker:=false', '-p', 'sparse_gp:=false'])
    node = pcm_node()
    try:
        fixture = runpy.run_path(str(Path(__file__).with_name('benchmark_pcm_sparse.py')))
        scenario, _, _ = fixture['scenario_inputs']('clear', 0.2)
        node.scenario_callback(scenario)
        node.ego_state.vx = 8.0
        node.powertrain_callback(ActuatorState(motor_speed_rad_s=225.0))
        node.last_exec = 0.0
        node.callback()
        stage = build_stage(node)[-1]
        initial = np.zeros(15)
        initial[4], initial[7] = 8.0, 225.0
        expected = node.rollout_eval(np.zeros(4), node.p_numeric)
        field = node.fsup_eval(np.zeros(4), node.p_numeric)[0].full().reshape(-1)
        for index in range(2):
            output = stage(initial, 8.0, node.p_numeric)
            np.testing.assert_allclose(float(output[0][4]), float(expected[1][index]), rtol=1e-9, atol=1e-8)
            np.testing.assert_allclose(output[4].full().reshape(-1), expected[0].full()[index], rtol=1e-9, atol=1e-8)
            np.testing.assert_allclose(float(output[2]), field[index], rtol=1e-9, atol=1e-8)
            initial = output[0].full().reshape(-1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


@pytest.mark.skipif(os.environ.get('PCM_ACADOS_STAGES_TEST') != '1', reason='Opt-in acados stage generation')
def test_native_acados_stage_adapter_and_rollout(tmp_path):
    import ctypes
    import runpy
    import rclpy
    from crp_msgs.msg import ActuatorState
    from plan_predictive_cost_map.pcm_longitudinal import pcm_node
    from plan_predictive_cost_map.pcm_acados_stages import export_stages

    rclpy.init(args=['--ros-args', '-p', 'prediction_model:=dynamic_actuators',
        '-p', 'lookahead_time:=0.2', '-p', 'command_lookahead_time:=0.1', '-p', 'prediction_steps:=2',
        '-p', 'actuator_solver.jit:=false', '-p', 'scenario_worker:=false', '-p', 'sparse_gp:=false'])
    node = pcm_node()
    try:
        fixture = runpy.run_path(str(Path(__file__).with_name('benchmark_pcm_sparse.py')))
        scenario, _, _ = fixture['scenario_inputs']('clear', 0.2)
        node.scenario_callback(scenario)
        node.ego_state.vx = 8.0
        node.powertrain_callback(ActuatorState(motor_speed_rad_s=225.0))
        node.last_exec = 0.0
        node.callback()
        model = export_stages(node, tmp_path)
        initial = np.zeros(15)
        initial[4], initial[7] = 8.0, 225.0
        commands = np.full(node.N, 8.0)
        sequence = [initial]
        for command in commands:
            sequence.append(model['stage'](sequence[-1], command, node.p_numeric)[0].full().reshape(-1))
        states = np.asfortranarray(np.column_stack(sequence))
        runtime = ctypes.CDLL(model['runtime_library'])
        runtime.pcm_stage_create.restype = ctypes.c_void_p
        pointer = ctypes.POINTER(ctypes.c_double)
        runtime.pcm_stage_solve.argtypes = [ctypes.c_void_p, pointer, pointer, pointer, pointer, ctypes.POINTER(ctypes.c_int)]
        runtime.pcm_stage_solve.restype = ctypes.c_int
        runtime.pcm_stage_free.argtypes = [ctypes.c_void_p]
        handle = runtime.pcm_stage_create()
        assert handle
        iterations = ctypes.c_int()
        try:
            status = runtime.pcm_stage_solve(handle, node.p_numeric.ctypes.data_as(pointer), initial.ctypes.data_as(pointer),
                commands.ctypes.data_as(pointer), states.ctypes.data_as(pointer), ctypes.byref(iterations))
        finally:
            runtime.pcm_stage_free(handle)
        assert status == 0, (status, iterations.value)
        assert np.isfinite(states).all() and np.isfinite(commands).all()
        replay = initial
        for index, command in enumerate(commands):
            output = model['stage'](replay, command, node.p_numeric)
            assert np.max(output[6].full()) <= 2 * node.solver_tolerance
            stage_residual = model['stage'](states[:, index], command, node.p_numeric)[0].full().reshape(-1) - states[:, index + 1]
            assert np.max(np.abs(stage_residual)) <= 2 * node.solver_tolerance
            replay = output[0].full().reshape(-1)
        print('Native acados stage commands:', commands, 'predicted speeds:', states[4], 'iterations:', iterations.value)
    finally:
        node.destroy_node()
        rclpy.shutdown()