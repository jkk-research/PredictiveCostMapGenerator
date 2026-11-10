import json
from pathlib import Path
import subprocess

import numpy as np
import rclpy
from rclpy.serialization import serialize_message
from autoware_perception_msgs.msg import ObjectClassification, PredictedObject, PredictedPath
from geometry_msgs.msg import Pose
from crp_msgs.msg import Ego

from plan_predictive_cost_map.pcm_cpp_model import export_model
from plan_predictive_cost_map.pcm_longitudinal import pcm_node


def test_native_parameter_pack_and_reporting_match_python(tmp_path):
    import runpy

    parameters = {'prediction_model': 'dynamic_actuators', 'lookahead_time': 0.5,
        'prediction_steps': 5, 'actuator_solver.jit': False, 'sparse_gp': False,
        'actuator_solver.method': 'sqp', 'scenario_worker': False}
    export_model(parameters, tmp_path)
    fixtures = runpy.run_path(str(Path(__file__).with_name('benchmark_pcm_sparse.py')))
    scenario, _, _ = fixtures['scenario_inputs']('clear', 0.5)
    actor = PredictedObject()
    actor.existence_probability = 0.8
    actor.classification = [ObjectClassification(label=7, probability=1.0)]
    actor.kinematics.initial_pose_with_covariance.pose.position.x = 12.0
    actor.kinematics.initial_pose_with_covariance.pose.position.y = -2.0
    actor.kinematics.initial_pose_with_covariance.pose.orientation.w = 1.0
    path = PredictedPath(confidence=1.0)
    path.time_step.nanosec = 125000000
    for position in (-2.0, -1.0, 0.0, 1.0, 2.0):
        pose = Pose()
        pose.position.x, pose.position.y, pose.orientation.w = 12.0, position, 1.0
        path.path.append(pose)
    actor.kinematics.predicted_paths = [path]
    scenario.local_moving_objects.objects = [actor]
    measured = Ego()
    measured.twist.twist.linear.x = 8.0
    measured.pose.pose.orientation.w = 1.0
    (tmp_path / 'scenario.cdr').write_bytes(serialize_message(scenario))
    (tmp_path / 'ego.cdr').write_bytes(serialize_message(measured))
    arguments = ['--ros-args']
    for key, value in parameters.items():
        arguments += ['-p', key + ':=' + json.dumps(value)]
    rclpy.init(args=arguments)
    node = pcm_node()
    try:
        node.ego_callback(measured)
        node.scenario_callback(scenario)
        pedestrian, pedestrian_alpha, _ = node.fit_dynamic_gp(node.pedestrians, node.max_pedestrians,
            node.p_ped, node.pedestrian_cost_primitive, node.pedestrian_noise_base, node.pedestrian_noise_growth)
        vehicle, vehicle_alpha, _ = node.fit_dynamic_gp(node.vehicles, node.max_vehicles,
            node.p_veh, node.vehicle_cost_primitive, node.vehicle_noise_base, node.vehicle_noise_growth)
        expected = np.r_[8.0, np.zeros(12), pedestrian_alpha, pedestrian.reshape(-1, order='F'),
            vehicle_alpha, vehicle.reshape(-1, order='F'), node.target_gp_points.reshape(-1, order='F'),
            node.target_gp_alpha, node.drivable_points.reshape(-1, order='F'), node.drivable_alpha,
            node.reference_data, np.zeros(17), 8.0 * 9.0 / 0.32, 0.0, 0.0, 0.0, node.target_speed]
        coefficients = [-0.5, 0.0, 0.0, 0.0]
        outputs = node.reporting_eval(coefficients, expected)
        root = Path(__file__).resolve().parents[3]
        executable = root / 'install/plan_predictive_cost_map/lib/plan_predictive_cost_map/pcm_longitudinal_cpp'
        completed = subprocess.run([str(executable), '--model-probe', str(tmp_path)],
            input=json.dumps({'coefficients': coefficients}), text=True, capture_output=True, check=True)
        result = json.loads(next(line.removeprefix('PARITY ') for line in completed.stdout.splitlines() if line.startswith('PARITY ')))
        np.testing.assert_allclose(result['parameters'], expected, rtol=1e-7, atol=1e-7)
        for actual, reference in zip(result['outputs'], outputs):
            np.testing.assert_allclose(actual, reference.full().reshape(-1, order='F'), rtol=1e-7, atol=1e-7)
    finally:
        node.destroy_node()
        rclpy.shutdown()