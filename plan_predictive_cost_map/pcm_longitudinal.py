#!/usr/bin/env python3

from timeit import default_timer
import json

import casadi as ca
import numpy as np
from scipy import interpolate
from scipy.ndimage import uniform_filter1d
from geometry_msgs.msg import Vector3

#import matplotlib.pyplot as plt
#import matplotlib as mpl
#from do_mpc.data import save_results

import rclpy
from rclpy.node import Node
from autoware_control_msgs.msg import Control
from crp_msgs.msg import Scenario
from autoware_planning_msgs.msg import TrajectoryPoint
from autoware_planning_msgs.msg import Trajectory
from crp_msgs.msg import Ego
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor
from dataclasses import dataclass, field, fields
import math
import matplotlib
matplotlib.use("TkAgg") # GUI backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from rclpy.executors import MultiThreadedExecutor
import threading
from plan_predictive_cost_map.pcm_kernels import lane_kernel, numeric_rbf_kernel, rbf_kernel
from plan_predictive_cost_map.pcm_lane_geometry import fit_drivable_gp, prepare_progress_reference
from plan_predictive_cost_map.pcm_longitudinal_dynamics import build_dynamic_light_rollout, build_dynamic_rollout, build_kinematic_rollout, dynamic_light_sample_times, shift_velocity_coefficients, velocity_polynomial, velocity_profile_coefficients
from plan_predictive_cost_map.planner.config import VehicleConfig
from plan_predictive_cost_map.pcm_scenario import (
    ObjectState, PredictionBranch, ScenarioSettings, parse_scenario, scenario_worker)
import multiprocessing
import queue
plt.ion() # interactive mode

@dataclass
class EgoState:
    vx: float
    ay: float
    ax: float
    yawrate: float
    steeringAngle: float
    vy: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0

class pcm_node(Node):
    def __init__(self) -> None:
        super().__init__("pcm_longitudinal")
        self.lanes = {"ego_lane_x": [], "ego_lane_y": [], "ego_lane_z": [], "left_lane_x": [], "left_lane_y": []}
        self.road_bounds = []
        self.scenario_stamp_ns = None
        self.oncomingObjects = [] # empty list
        self.followedObjects = [] # empty list
        self.vehicles = []
        self.pedestrians = []

        self.time_history = []
        self.c_el_history = []

        # 1. Initialize Horizon and Symbols
        self.prediction_model = self.declare_parameter(
            'prediction_model', 'dynamic', ParameterDescriptor(read_only=True,
                description='Prediction model selected at startup: kinematic, dynamic_light or dynamic')).value
        if self.prediction_model not in ('kinematic', 'dynamic_light', 'dynamic'):
            raise ValueError('prediction_model must be kinematic, dynamic_light or dynamic')
        self.tau_h = self.declare_parameter('lookahead_time', 4.0).value
        self.N = self.declare_parameter('prediction_steps', 20).value
        self.timer_period = self.declare_parameter('timer_period', 0.05).value
        self.target_speed = 0.0
        self.command_lookahead_time = self.declare_parameter('command_lookahead_time', 0.25).value
        self.lane_noise_std = self.declare_parameter('lane_noise_std', 0.001).value
        self.reference_samples = self.declare_parameter('reference_samples', 96).value
        self.target_support_points = self.declare_parameter('target_support_points', 128).value
        if self.reference_samples < 8 or self.target_support_points < 4:
            raise ValueError('Reference and target GP support budgets are too small')
        self.reference_data = prepare_progress_reference(np.array([[0., 0., 0.], [1., 0., 0.]]), self.reference_samples)
        self.target_gp_points = np.full((self.target_support_points, 2), 1000.0)
        self.target_gp_alpha = np.zeros(self.target_support_points)
        self.drivable_support_points = self.declare_parameter('drivable_support_points', 128).value
        self.drivable_noise_std = self.declare_parameter('drivable_noise_std', 0.05).value
        if (self.drivable_support_points < 4
                or not math.isfinite(self.drivable_noise_std) or self.drivable_noise_std <= 0.0):
            raise ValueError('Invalid lane geometry support budget or road noise')
        self.drivable_points = np.full((self.drivable_support_points, 2), 1000.0)
        self.drivable_alpha = np.zeros(self.drivable_support_points)
        self.pedestrian_noise_base = self.declare_parameter('pedestrian_noise_base', 0.02).value
        self.pedestrian_noise_growth = self.declare_parameter('pedestrian_noise_growth', 0.25).value
        self.vehicle_noise_base = self.declare_parameter('vehicle_noise_base', 0.01).value
        self.vehicle_noise_growth = self.declare_parameter('vehicle_noise_growth', 0.05).value
        self.solver_tolerance = self.declare_parameter('solver_tolerance', 0.001).value
        self.max_gp_cost = self.declare_parameter('max_gp_cost', 0.05).value
        if not math.isfinite(self.max_gp_cost):
            raise ValueError('max_gp_cost must be finite')
        self.publish_gp_state = self.declare_parameter('publish_gp_state', False).value
        self.gp_state_period = self.declare_parameter('gp_state_period', 0.25).value
        self.last_gp_state_time = -math.inf
        if self.N < 2:
            raise ValueError('prediction_steps must be at least 2')
        for name, value in (
            ('lookahead_time', self.tau_h), ('timer_period', self.timer_period),
            ('lane_noise_std', self.lane_noise_std),
            ('pedestrian_noise_base', self.pedestrian_noise_base),
            ('vehicle_noise_base', self.vehicle_noise_base), ('solver_tolerance', self.solver_tolerance),
            ('gp_state_period', self.gp_state_period),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        for name, value in (('pedestrian_noise_growth', self.pedestrian_noise_growth),
                            ('vehicle_noise_growth', self.vehicle_noise_growth)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not math.isfinite(self.command_lookahead_time) or not 0.0 <= self.command_lookahead_time <= self.tau_h:
            raise ValueError('command_lookahead_time must lie within the lookahead_time')
        self.dt = self.tau_h / self.N
        self.t = ca.vertcat(*[self.dt * (i+1) for i in range(self.N)])        
        self.M = 3 if self.prediction_model == 'kinematic' else 4
        defaults = VehicleConfig()
        self.vehicle_settings = VehicleConfig(**{
            setting.name: self.declare_parameter('vehicle.' + setting.name, getattr(defaults, setting.name)).value
            for setting in fields(defaults)})
        self.model_control_step = self.declare_parameter('vehicle.control_step', 0.05).value
        self.use_road_gradient = self.declare_parameter('vehicle.use_road_gradient', True).value
        self.model_road_pitch = self.declare_parameter('vehicle.road_pitch', 0.0).value
        self.light_constraint_step = self.declare_parameter('dynamic_light.constraint_step', 0.1).value
        if (not math.isfinite(self.model_control_step) or self.model_control_step <= 0.0
                or (self.prediction_model != 'kinematic' and self.model_control_step > self.dt)
                or not math.isfinite(self.light_constraint_step) or self.light_constraint_step <= 0.0
                or (self.prediction_model == 'dynamic_light' and self.light_constraint_step > self.dt)
                or not math.isfinite(self.model_road_pitch) or abs(self.model_road_pitch) >= math.pi / 2.0):
            raise ValueError('Invalid vehicle control step, light constraint step or road pitch')
        if self.prediction_model == 'dynamic':
            self.model_rollout = build_dynamic_rollout(
                self.tau_h, self.N, self.model_control_step, self.command_lookahead_time,
                self.vehicle_settings, self.use_road_gradient, self.model_road_pitch, self.reference_samples)
        elif self.prediction_model == 'dynamic_light':
            self.model_rollout = build_dynamic_light_rollout(
                self.tau_h, self.N, self.light_constraint_step, self.command_lookahead_time,
                self.vehicle_settings, self.use_road_gradient, self.model_road_pitch, self.reference_samples)
        else:
            self.model_rollout = build_kinematic_rollout(
                self.tau_h, self.N, self.command_lookahead_time, self.vehicle_settings, self.reference_samples)

        self.ego_state = EgoState(13.0,0,0,0,0)

        self.c_el = [0, 0, 0, 0]
        self.c_ez = np.zeros(4)
        self.c_ol = [0, 0, 0, 0]
        self.lane_received = False

        for attribute, name, defaults in (
            ('p_ped', 'pedestrian_kernel', [0.0, math.log(5.0), math.log(1.5), math.log(0.5)]),
            ('p_veh', 'vehicle_kernel', [-1.0, 1.0, math.log(2.2), 0.0]),
            ('p_el', 'ego_lane_kernel', [-1.0, 1.0, 0.0]),
            ('p_ol', 'opposite_lane_kernel', [-1.0, 1.0, math.log(2.0)]),
        ):
            values = list(self.declare_parameter(name, [math.exp(value) for value in defaults]).value)
            if len(values) != len(defaults) or any(not math.isfinite(value) or value <= 0.0 for value in values):
                dimensions = 'scale, length_x, length_y' + (', length_time' if len(defaults) == 4 else '')
                raise ValueError(f'{name} must contain {len(defaults)} finite positive values: {dimensions}')
            setattr(self, attribute, values)


        self.ctrl_cmd = Control()
        self.trajectory = Trajectory()

        self.runningTime = 0.0
        self.plot_results = False
        self.max_pedestrians = self.declare_parameter('max_pedestrians', 2).value
        self.max_vehicles = self.declare_parameter('max_vehicles', 2).value
        self.max_predicted_paths = self.declare_parameter('max_predicted_paths', 2).value
        self.dynamic_gp_state = {}
        self.ay_max = self.declare_parameter('max_lateral_acceleration', 3.0).value
        self.ax_max = self.declare_parameter('max_longitudinal_acceleration', 5.0).value
        if self.max_pedestrians < 1 or self.max_vehicles < 1 or self.max_predicted_paths < 1:
            raise ValueError('object and predicted-path limits must be at least 1')
        self.pedestrian_support_count = self.max_pedestrians * self.max_predicted_paths * self.N
        self.vehicle_support_count = self.max_vehicles * self.max_predicted_paths * self.N
        if any(not math.isfinite(value) or value <= 0.0 for value in (self.ay_max, self.ax_max)):
            raise ValueError('acceleration limits must be finite and positive')
        self.pedestrian_cost_primitive = self.declare_parameter('pedestrian_cost', 3.0).value
        self.vehicle_cost_primitive = self.declare_parameter('vehicle_cost', 1.0).value
        self.opplane_cost_primitive = self.declare_parameter('opposite_lane_cost', -0.4).value
        self.egolane_cost_primitive = self.declare_parameter('ego_lane_cost', -1.0).value
        if any(not math.isfinite(value) for value in (self.pedestrian_cost_primitive, self.vehicle_cost_primitive,
                                                     self.opplane_cost_primitive, self.egolane_cost_primitive)):
            raise ValueError('cost primitives must be finite')
        if self.pedestrian_cost_primitive < 0.0 or self.vehicle_cost_primitive < 0.0:
            raise ValueError('dynamic-object severity must be nonnegative')
        self.scenario_settings = ScenarioSettings(
            self.dt, self.reference_samples, self.target_support_points, self.drivable_support_points,
            tuple(self.p_el), tuple(self.p_ol), self.egolane_cost_primitive, self.opplane_cost_primitive,
            self.lane_noise_std, self.drivable_noise_std)
        self.scenario_process = None

        self.replan = True
        self.replan_timer = 0.0
        self.c_numeric = np.zeros(self.M)

        # symbolic definitions and Casadi optimization setup 
        self.c_sym = ca.MX.sym('c', self.M)              

        self.z_pedestrians_sym = ca.MX.sym('z_pedestrians_sym', self.pedestrian_support_count, 3)
        self.z_vehicles_sym = ca.MX.sym('z_vehicles_sym', self.vehicle_support_count, 3)
        self.dynamic_context_sym = ca.MX.sym('dynamic_context', 13)
        self.reference_sym = ca.MX.sym('progress_reference_data', 2 + 6 * self.reference_samples)
        self.target_gp_points_sym = ca.MX.sym('target_gp_points', self.target_support_points, 2)
        self.target_gp_alpha_sym = ca.MX.sym('target_gp_alpha', self.target_support_points)
        self.drivable_points_sym = ca.MX.sym('drivable_points', self.drivable_support_points, 2)
        self.drivable_alpha_sym = ca.MX.sym('drivable_alpha', self.drivable_support_points)
        self.p_sym = ca.vertcat(ca.MX.sym('v_0'), ca.MX.sym('c_ego', 4, 1), ca.MX.sym('c_el', 4, 1), ca.MX.sym('c_ol', 4, 1), ca.MX.sym('alpha_ped', self.pedestrian_support_count, 1), ca.reshape(self.z_pedestrians_sym, -1, 1),
                                ca.MX.sym('alpha_veh', self.vehicle_support_count, 1), ca.reshape(self.z_vehicles_sym, -1, 1),
                                ca.reshape(self.target_gp_points_sym, -1, 1), self.target_gp_alpha_sym,
                                ca.reshape(self.drivable_points_sym, -1, 1), self.drivable_alpha_sym,
                                self.reference_sym,
                                self.dynamic_context_sym, ca.MX.sym('target_speed'))

        context = self.dynamic_context_sym
        initial = ca.vertcat(context[5:9], self.p_sym[0], context[0:2])
        light = self.prediction_model == 'dynamic_light'
        prediction = self.model_rollout.call([self.c_sym, initial, context[3], context[4],
                                              self.p_sym[1:5], context[9:13], context[2], self.reference_sym],
                                             light, False)
        shared_prediction = prediction if light else None
        self.rollout_eval = ca.Function(self.prediction_model + '_prediction', [self.c_sym, self.p_sym], list(prediction))
        positions, speeds, accelerations, requested_acceleration, command_speed, steering, dense_requests = prediction[:7]
        self.command_eval = ca.Function(self.prediction_model + '_velocity_command', [self.c_sym, self.p_sym], [command_speed])

        # Constraints (Kinematics)
        # acceleration
        g_ax = accelerations[:, 0]
        g_ay = accelerations[:, 1]
        g_vx = speeds

        g_ax_upper = g_ax - self.ax_max
        g_ax_lower = - self.ax_max - g_ax

        g_ay_upper = g_ay - self.ay_max
        g_ay_lower = - self.ay_max - g_ay

        g_vx_lower = -g_vx
        g_vx_upper = g_vx - ca.fmax(self.p_sym[-1], self.p_sym[0])

        # Safety cost constraint
        cost_safety, fsup_vec = self.cost_sup(self.c_sym, self.p_sym, prediction=shared_prediction)

        # Stack all nonlinear constraints
        #self.g = g_vx_lower
        self.g =  ca.vertcat(
            g_ax_upper,
            g_ax_lower,
            g_vx_lower,
            g_vx_upper,
            fsup_vec - self.max_gp_cost,
            g_ay_upper,
            g_ay_lower,
        )
        if self.prediction_model != 'kinematic':
            self.g = ca.vertcat(self.g, dense_requests - self.vehicle_settings.maximum_acceleration,
                                -self.vehicle_settings.maximum_deceleration - dense_requests)
        if self.prediction_model == 'dynamic_light':
            dense_times = ca.DM(dynamic_light_sample_times(self.tau_h, self.N, self.light_constraint_step,
                                                          self.command_lookahead_time)[1:])
            dense_speed = velocity_polynomial(dense_times, self.c_sym, self.p_sym[0], self.tau_h)[0]
            self.g = ca.vertcat(self.g, -dense_speed,
                                dense_speed - ca.fmax(self.p_sym[-1], self.p_sym[0]))
        self.g = ca.vertcat(self.g, prediction[9] - self.reference_sym[1])

        # 1. Get the individual cost terms (Symbolic)
        cost_velocity = self.get_cost_vel(self.c_sym, self.p_sym, prediction=shared_prediction) # Range [0, ~1]
        cost_ay = ca.sumsqr(g_ay)/(self.ay_max**2 * self.N)
        cost_ax = ca.sumsqr(g_ax)/(self.ax_max * self.N)
        # collision penalty only
        cost_crash = ca.sum1(
            ca.exp(5*self.softplus(fsup_vec))-1
        )

        # 2. Define Weights
        # weight_safety: Keep the car away from objects/lane bounds
        # weight_vel: Keep the car at the target speed
        w_safety = self.declare_parameter('weight_safety', 0.9).value
        w_vel = self.declare_parameter('weight_velocity', 0.5).value
        w_ay = self.declare_parameter('weight_lateral_acceleration', 0.5).value
        w_ax = self.declare_parameter('weight_acceleration', 0.0).value
        w_crash = self.declare_parameter('weight_crash', 1.0).value
        if any(not math.isfinite(value) or value < 0.0 for value in (w_safety, w_vel, w_ay, w_ax, w_crash)):
            raise ValueError('objective weights must be finite and nonnegative')
        
        # 3. Combine into final Objective
        # We also keep the regularization for steering smoothness (c_lat)
        obj = (w_safety * cost_safety) + (w_vel * cost_velocity) + (w_ay * cost_ay) + (w_ax * cost_ax) + (w_crash * cost_crash)

        # 6. Setup and Solve NLP
        nlp = {'x': self.c_sym, 'f': obj, 'g': self.g, 'p': self.p_sym}
        opts = {
            'ipopt.print_level':0,
            'print_time':0,
            'ipopt.tol':self.solver_tolerance,
            'ipopt.warm_start_init_point':'yes',
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        if light:
            # A feasible shifted plan is already near the optimum, so start with a small barrier.
            self.light_warm_solver = ca.nlpsol('light_warm_solver', 'ipopt', nlp, {
                **opts, 'ipopt.mu_init': 1e-3, 'ipopt.warm_start_bound_push': 1e-6,
                'ipopt.warm_start_slack_bound_push': 1e-6, 'ipopt.warm_start_mult_bound_push': 1e-6})
            self.nlp_objective = self.solver.get_function('nlp_f')
            self.nlp_constraints = self.solver.get_function('nlp_g')
        self.light_previous = None
        self.light_previous_time = 0.0
        self.light_multipliers = {}
        self.light_warm_start_violation = 0.05

        x_traj = positions[:, 0]
        y_traj = positions[:, 1]
        vx_traj = speeds

        self.traj_eval = ca.Function(
            'traj_eval',
            [self.c_sym, self.p_sym],
            [
                x_traj,
                y_traj,
                vx_traj
            ]
        )

        # Evaluate f_sup_vec (= g_cost) numerically after optimization
        _, fsup_vec, *gp_state = self.cost_sup(self.c_sym, self.p_sym, include_gp_state=True)

        self.fsup_eval = ca.Function(
            'fsup_eval',
            [self.c_sym, self.p_sym],
            [fsup_vec, *gp_state]
)   
        

        # create publishers/subscribers
        self.get_logger().info(f'Prediction model: {self.prediction_model}')
        self.timer = self.create_timer(self.timer_period, self.callback)

        self.cmd_pub = self.create_publisher(Control, '/control/command/control_cmd', 30)

        self.trajectory_pub = self.create_publisher(Trajectory, '/plan/trajectory', 30)
        self.gp_state_pub = self.create_publisher(
            String, '/plan/pcm_longitudinal/gp_state',
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))

        if self.declare_parameter('scenario_worker', True).value:
            # A separate process keeps message deserialization and GP preprocessing off the planning thread.
            context = multiprocessing.get_context('spawn')
            self.scenario_requests = context.Queue()
            self.scenario_results = context.Queue()
            self.scenario_process = context.Process(
                target=scenario_worker, args=(self.scenario_requests, self.scenario_results, self.scenario_settings),
                daemon=True)
            self.scenario_process.start()
            self.traj_sub = self.create_subscription(
                Scenario, '/scenario', self.scenario_requests.put, 1, raw=True)
        else:
            self.traj_sub = self.create_subscription(
                Scenario, '/scenario', self.scenario_callback, 1
                )
        
        self.ego_sub = self.create_subscription(
            Ego, '/ego', self.ego_callback, 1
            )

    def softplus(self, x, beta=10):
        return ca.log(1 + ca.exp(beta*x))/beta

    def ego_callback(self, input_msg):
        pose = input_msg.pose.pose
        rotation = pose.orientation
        pitch = -math.asin(np.clip(2.0 * (rotation.w * rotation.y - rotation.z * rotation.x), -1.0, 1.0))
        self.ego_state = EgoState(
            vx=input_msg.twist.twist.linear.x, ay=input_msg.accel.accel.linear.y,
            ax=input_msg.accel.accel.linear.x, yawrate=input_msg.twist.twist.angular.z,
            steeringAngle=input_msg.tire_angle_front, vy=input_msg.twist.twist.linear.y,
            x=pose.position.x, y=pose.position.y, z=pose.position.z,
            yaw=self.quaternion_to_yaw(rotation), pitch=pitch)

    def scenario_callback(self, input_msg):
        self.apply_scenario(parse_scenario(input_msg, self.scenario_settings))

    def poll_scenario_updates(self):
        update = None
        while self.scenario_process is not None:
            try:
                update = self.scenario_results.get_nowait()
            except queue.Empty:
                break
        if update is not None:
            self.apply_scenario(update)

    def apply_scenario(self, update):
        if 'error' in update:
            self.get_logger().error(f"Scenario preprocessing failed: {update['error']}", throttle_duration_sec=2.0)
            return
        for warning in update['warnings']:
            self.get_logger().warn(warning)
        self.scenario_stamp_ns = update['stamp_ns']
        self.pedestrians = update['pedestrians']
        self.vehicles = update['vehicles']
        self.lanes = update['lanes']
        self.road_bounds = update['road_bounds']
        self.target_speed = update['target_speed']
        self.lane_received = update['lane_received']
        self.c_el = np.zeros(4)
        self.c_ez = np.zeros(4)
        self.c_ol = np.zeros(4)
        if update['reference_data'] is not None:
            self.reference_data = update['reference_data']
        if update['target_gp'] is not None:
            self.target_gp_points, self.target_gp_alpha = update['target_gp']
        self.drivable_points, self.drivable_alpha = update['drivable_gp']

    def destroy_node(self):
        if self.scenario_process is not None and self.scenario_process.is_alive():
            self.scenario_requests.put(None)
            self.scenario_process.join(timeout=2.0)
            if self.scenario_process.is_alive():
                self.scenario_process.terminate()
        return super().destroy_node()

    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)  

    def k(self, X, Z, p):
        return rbf_kernel(X, Z, p)

    # In f_dyn
    def f_dyn(self, X, Z, y, p, sigma_n, alpha):
        return (self.k(X, Z, p) @ alpha) - 1.0

    def f_dyn_(self, X, Z, y, p, sigma_n, alpha):      
        return lane_kernel(X, Z, p) @ alpha
    
    def smooth_max(self,a,b,eps=0.01):
        return 0.5*(a+b+ca.sqrt((a-b)**2+eps))

    def f_sup(self, X,
                    Z_ped, y_ped, p_ped,
                    Z_veh, y_veh, p_veh,
                    Z_el, y_el, p_el,
                    Z_ol, y_ol, p_ol,
                    alpha_el, alpha_ol, alpha_ped, alpha_veh):
        avg_dyn = self.smooth_max( self.f_dyn(X, Z_ped, y_ped, p_ped, 1, alpha_ped),
                        self.f_dyn(X, Z_veh, y_veh, p_veh, 1, alpha_veh))

        min_dyn = ca.fmin(self.f_dyn_(X, Z_el, y_el, p_el, 1, alpha_el),
                        self.f_dyn_(X, Z_ol, y_ol, p_ol, 1, alpha_ol))
        
        return self.smooth_max(avg_dyn,min_dyn)

    def cost_sup(self, c, p, include_gp_state=False, prediction=None):

        # get the parameters from the concatenated vector
        alpha_ped = p[13:13 + self.pedestrian_support_count]
        z_pedestrians_flat = p[13 + self.pedestrian_support_count:13 + 4 * self.pedestrian_support_count]
        vehicle_offset = 13 + 4 * self.pedestrian_support_count
        alpha_veh = p[vehicle_offset:vehicle_offset + self.vehicle_support_count]
        z_vehicles_flat = p[vehicle_offset + self.vehicle_support_count:
                    vehicle_offset + 4 * self.vehicle_support_count]
        geometry_offset = vehicle_offset + 4 * self.vehicle_support_count
        target_end = geometry_offset + 2 * self.target_support_points
        target_gp_points = ca.reshape(p[geometry_offset:target_end], self.target_support_points, 2)
        target_alpha_end = target_end + self.target_support_points
        target_gp_alpha = p[target_end:target_alpha_end]
        road_end = target_alpha_end + 2 * self.drivable_support_points
        road_points = ca.reshape(p[target_alpha_end:road_end], self.drivable_support_points, 2)
        road_alpha = p[road_end:road_end + self.drivable_support_points]

        # 1. Generate Ego Trajectory (Shape: N x 3)
        if prediction is None:
            prediction = self.rollout_eval(c, p)
        x_ego = prediction[0][:, 0]
        y_ego = prediction[0][:, 1]
        # We must transpose to get [N x 3] so it matches the Kernel 'X' logic
        ego_traj = ca.horzcat(x_ego, y_ego, self.t)

        # 2. Generate Inducing Points for Lanes (Inducing points along ego x-path)
        # These create the "Safe/Unsafe" zones for the GP
        
        ego_lane_pts = target_gp_points
        opp_lane_pts = road_points

        # Generate inducing points for the lanes
        # Ego lane
        alpha_el = target_gp_alpha

        # Oncoming lane
        alpha_ol = road_alpha

        # Generate inducing points for pedestrians
        
        Z_ped = ca.reshape(
            z_pedestrians_flat,
            self.pedestrian_support_count,
            3
        )
        y_ped = np.zeros((self.pedestrian_support_count, 1))

        # Generate inducing points for vehicles

        Z_veh = ca.reshape(
            z_vehicles_flat,
            self.vehicle_support_count,
            3
        )
        y_veh = np.zeros((self.vehicle_support_count, 1))

        # 3. Evaluate f_sup
        # This returns an [N x 1] vector of costs (one for each time step)
        fsup_vec = self.f_sup(
            ego_traj,
            Z_ped, y_ped, self.p_ped,
            Z_veh, y_veh, self.p_veh,
            ego_lane_pts, ca.DM.ones(self.target_support_points, 1) * self.egolane_cost_primitive, self.p_el, # Ego lane
            opp_lane_pts, ca.DM.ones(self.drivable_support_points, 1) * self.opplane_cost_primitive, self.p_ol,  # Opp lane
            alpha_el, alpha_ol, alpha_ped, alpha_veh # pre-computed symbolix kernel
        )

        # 4. Convert vector to scalar for optimization
        # Option A: Average cost (smoother gradient)
        # Option B: ca.sum1(ca.fmax(0, fsup_vec)) (penalize only high cost)
        total_cost = ca.sumsqr(fsup_vec + 1.0) / self.N
        if include_gp_state:
            return total_cost, fsup_vec, ego_lane_pts, opp_lane_pts, alpha_el, alpha_ol
        return total_cost, fsup_vec


    def get_cost_vel(self, c_sym, p_sym, prediction=None):
        target_v = p_sym[-1]
        vx_profile = (self.rollout_eval(c_sym, p_sym) if prediction is None else prediction)[1]
        
        # Calculate error
        vel_error = target_v - vx_profile
        
        # MATLAB: (err' * err) / ( (tau_h/dt) * target_vel^2 )
        # ca.sumsqr(x) is equivalent to x' * x
        denominator = self.N * ca.fmax(target_v**2, 1.0)
        
        return ca.sumsqr(vel_error) / denominator


    def sample_object_path(self, obj, branch=None):
        times = np.asarray(self.t).flatten()
        prediction = obj if branch is None else branch
        if len(prediction.predicted_path) == 0:
            return np.column_stack((obj.x + obj.vx * np.cos(obj.theta) * times,
                                    obj.y + obj.vx * np.sin(obj.theta) * times))
        step = prediction.prediction_time_step if prediction.prediction_time_step > 0.0 else self.dt
        sample_times = np.arange(len(prediction.predicted_path)) * step
        return np.column_stack((np.interp(times, sample_times, prediction.predicted_path[:, 0]),
                                np.interp(times, sample_times, prediction.predicted_path[:, 1])))

    def fit_dynamic_gp(self, objects, max_objects, kernel, severity, noise_base, noise_growth):
        count = max_objects * self.max_predicted_paths * self.N
        points = np.full((count, 3), 1000.0)
        targets = np.zeros(count)
        noise = np.full(count, noise_base)
        existence = np.zeros(count)
        confidence = np.zeros(count)
        times = np.asarray(self.t).flatten()
        used = 0
        dropped_branches = 0
        for obj in objects[:max_objects]:
            if obj.existence_probability <= 0.0:
                continue
            branches = obj.predicted_branches or [PredictionBranch(
                obj.predicted_path, obj.prediction_time_step, obj.trajectory_confidence)]
            branches = sorted((branch for branch in branches if branch.confidence > 0.0),
                              key=lambda branch: branch.confidence, reverse=True)
            dropped_branches += max(0, len(branches) - self.max_predicted_paths)
            for branch in branches[:self.max_predicted_paths]:
                section = slice(used, used + self.N)
                points[section] = np.column_stack((self.sample_object_path(obj, branch), times))
                existence[section] = np.clip(obj.existence_probability, 0.0, 1.0)
                confidence[section] = np.clip(branch.confidence, 0.0, 1.0)
                probability = existence[section] * confidence[section]
                targets[section] = 1.0 + probability * severity
                noise[section] = (noise_base + noise_growth * times
                                  + probability * (1.0 - probability) * severity**2)
                used += self.N
        if len(objects) > max_objects or dropped_branches:
            self.get_logger().warn(
                f'GP capacity reached: {max(0, len(objects) - max_objects)} objects and '
                f'{dropped_branches} prediction branches omitted', throttle_duration_sec=5.0)
        matrix = numeric_rbf_kernel(points, points, kernel)
        alpha = np.linalg.solve(matrix + np.diag(noise), targets)
        metadata = {'targets': targets.tolist(), 'noise_variances': noise.tolist(),
                    'existence_probability': existence.tolist(), 'trajectory_confidence': confidence.tolist()}
        return points, alpha, metadata

    def callback(self):
        t1 = default_timer()   

        now = t1
        self.last_exec = getattr(self, 'last_exec', now)

        # Tolerate timer jitter; an early tick would otherwise skip a whole period.
        if now - self.last_exec < 0.5 * self.timer_period:
            return

        self.last_exec = now
        self.poll_scenario_updates()

        # Initial Guess (Warm start)
        if not hasattr(self, 'last_c_opt'): self.last_c_opt = np.zeros(self.M)    
    
        # Ensure we have lane data before proceeding
        if not self.lane_received:
            self.get_logger().warn("Waiting for lane data...")
            return
        
        p_obj, alpha_ped, pedestrian_state = self.fit_dynamic_gp(
            self.pedestrians, self.max_pedestrians, self.p_ped, self.pedestrian_cost_primitive,
            self.pedestrian_noise_base, self.pedestrian_noise_growth)
        p_veh, alpha_veh, vehicle_state = self.fit_dynamic_gp(
            self.vehicles, self.max_vehicles, self.p_veh, self.vehicle_cost_primitive,
            self.vehicle_noise_base, self.vehicle_noise_growth)
        self.dynamic_gp_state = {'pedestrian': pedestrian_state, 'vehicle': vehicle_state}

        c_ego = self.c_el

        self.p_numeric = np.concatenate([
            np.array([self.ego_state.vx]), # 0
            c_ego, # 1-4
            self.c_el, # 5-8
            self.c_ol, # 9-12
            alpha_ped.reshape(-1), #13:N*max_ped
            p_obj.reshape(-1, order='F'),
            alpha_veh.reshape(-1), #13:N*max_ped
            p_veh.reshape(-1, order='F'),
            self.target_gp_points.reshape(-1, order='F'), self.target_gp_alpha,
            self.drivable_points.reshape(-1, order='F'), self.drivable_alpha,
            self.reference_data,
            np.array([self.ego_state.vy, self.ego_state.yawrate, self.ego_state.pitch,
                      self.ego_state.ax, self.ego_state.steeringAngle, self.ego_state.x,
                      self.ego_state.y, self.ego_state.z, self.ego_state.yaw, *self.c_ez]),
            np.array([self.target_speed]),
        ]).flatten()

        if self.replan:
            # Variable Bounds (lbx, ubx)
            lbc = -np.inf * np.ones(self.M)
            ubc =  np.inf * np.ones(self.M)

            n_g = self.g.numel()

            lbg = -np.inf * np.ones(n_g)
            ubg = np.zeros(n_g)

            if not hasattr(self, 'last_lam_g'):
                self.last_lam_g = np.zeros(n_g)

            if not hasattr(self, 'last_lam_x'):
                self.last_lam_x = np.zeros(self.M)

            try:
                if self.prediction_model == 'dynamic_light':
                    solver, initial, multipliers, initialization = self.light_initial_guess(t1)
                else:
                    solver, initial, initialization = self.solver, self.last_c_opt, 'previous'
                    multipliers = {'lam_x0': self.last_lam_x, 'lam_g0': self.last_lam_g}
                sol = solver(x0=initial, lbx=lbc, ubx=ubc, lbg=lbg, ubg=ubg, p=self.p_numeric, **multipliers)
                stats = solver.stats()
                candidate = sol['x'].full().flatten()
                constraints = sol['g'].full().flatten()
                valid = (stats.get('success', False) and candidate.shape == (self.M,) and np.isfinite(candidate).all()
                         and np.isfinite(constraints).all() and math.isfinite(float(sol['f']))
                         and np.max(constraints) <= max(1e-6, self.solver_tolerance * 2.0))
                self.plan_diagnostics = {'status': stats.get('return_status', ''), 'accepted': bool(valid),
                                         'prediction_model': self.prediction_model,
                                         'maximum_constraint': (float(np.max(constraints))
                                                                if np.isfinite(constraints).all() else None),
                                         'ego_speed': float(self.ego_state.vx),
                                         'initialization': initialization,
                                         'iterations': int(stats.get('iter_count', -1)),
                                         'solve_seconds': default_timer() - t1}
            except RuntimeError as error:
                self.plan_diagnostics = {'status': str(error), 'accepted': False,
                                         'prediction_model': self.prediction_model,
                                         'ego_speed': float(self.ego_state.vx)}
                self.reject_solver_solution(str(error))
                return
            if not valid:
                self.reject_solver_solution(stats.get('return_status', 'Invalid solver output'))
                return
            self.c_numeric = sol['x']
            self.last_c_opt = self.c_numeric.full().flatten()
            self.last_lam_x = sol['lam_x'].full()
            self.last_lam_g = sol['lam_g'].full()
            if self.prediction_model == 'dynamic_light':
                self.light_previous = self.last_c_opt.copy()
                self.light_previous_time = t1
                self.light_multipliers = {'lam_x0': sol['lam_x'], 'lam_g0': sol['lam_g']}

            self.get_logger().info(
                f"status: {stats['return_status']}, "
                f"iterations: {stats['iter_count']}"
            )

            self.replan = False
            self.replan_timer += self.timer_period               

        # Evaluate f_sup_vec (= g_cost) for the optimized solution
        gp_evaluation = self.fsup_eval(self.c_numeric, self.p_numeric)
        f_sup_vec_numeric = gp_evaluation[0]

        # Convert CasADi DM -> NumPy array for printing
        f_sup_vec_numeric = np.array(f_sup_vec_numeric.full()).flatten()

        if np.any(f_sup_vec_numeric > 0.0) or self.replan_timer > 0.0: 
            self.replan = True
            self.replan_timer = 0.0
        else:
            self.replan = False
            self.replan_timer += self.timer_period
            self.get_logger().info(
                f"Maximum value: {np.max(f_sup_vec_numeric)}"
            )

        self.get_logger().info(f"f_sup_vec: {f_sup_vec_numeric}")
        self.get_logger().info(f"replan_timer: {self.replan_timer:.2f}s, replan: {self.replan}")

        prediction = self.rollout_eval(self.c_numeric, self.p_numeric)
        velocityTarget = max(float(prediction[4]), 0.0)

        self.get_logger().info(
            f"ego_vx={self.ego_state.vx}, "
            f"target_vx={velocityTarget}, "
            f"replan={self.replan}"
        )
        
        control_cmd_msg = Control()
        control_cmd_msg.stamp = self.get_clock().now().to_msg()
        control_cmd_msg.longitudinal.velocity = float(velocityTarget)
        self.cmd_pub.publish(control_cmd_msg)

        # getting the trajectory
        x_final = prediction[0][:, 0].full().flatten()
        y_final = prediction[0][:, 1].full().flatten()
        vx_final = prediction[1].full().flatten()
        ax_final = prediction[2][:, 0].full().flatten()
        steering_final = prediction[5].full().flatten()
        orientation_final = prediction[7].full()
        lateral_final = prediction[8][:, 5].full().flatten()
        yawrate_final = prediction[8][:, 6].full().flatten()
        z_final = prediction[0][:, 2].full().flatten()


        # publishing the trajectory
        trajectory_msg = Trajectory()
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        trajectory_msg.header.frame_id = "base_link"

        for i in range(len(x_final)):
            pt = TrajectoryPoint()

            # position
            pt.pose.position.x = float(x_final[i])
            pt.pose.position.y = float(y_final[i])
            pt.pose.position.z = float(z_final[i])
            pt.pose.orientation.x = float(orientation_final[i, 0])
            pt.pose.orientation.y = float(orientation_final[i, 1])
            pt.pose.orientation.z = float(orientation_final[i, 2])
            pt.pose.orientation.w = float(orientation_final[i, 3])

            # velocity      
            pt.longitudinal_velocity_mps = float(vx_final[i])
            pt.acceleration_mps2 = float(ax_final[i])
            pt.front_wheel_angle_rad = float(steering_final[i])
            pt.lateral_velocity_mps = float(lateral_final[i])
            pt.heading_rate_rps = float(yawrate_final[i])
            elapsed = self.dt * (i + 1)
            pt.time_from_start.sec = int(elapsed)
            pt.time_from_start.nanosec = int(round((elapsed - int(elapsed)) * 1e9))

            # append point
            trajectory_msg.points.append(pt)

        # publish
        self.trajectory_pub.publish(trajectory_msg)
        self.publish_gp_snapshot(gp_evaluation[1:])
        
        t_end = default_timer()
        self.get_logger().info(f"Loop time: {t_end-t1:.4f}s")

    def light_initial_guess(self, now):
        # Initialization only: every published plan must still be an accepted IPOPT solution.
        guesses = []
        if self.light_previous is not None:
            delta = min(max(now - self.light_previous_time, 0.0), self.tau_h)
            shifted = shift_velocity_coefficients(self.light_previous, delta, self.tau_h)
            violation = float(np.max(self.nlp_constraints(shifted, self.p_numeric).full()))
            if violation <= self.light_warm_start_violation:
                return self.light_warm_solver, shifted, self.light_multipliers, 'shifted'
            guesses.append(shifted)
        guesses += velocity_profile_coefficients(self.tau_h, max(float(self.ego_state.vx), 0.0), self.target_speed)
        scores = [(max(float(np.max(self.nlp_constraints(guess, self.p_numeric).full())), 0.0),
                   float(self.nlp_objective(guess, self.p_numeric))) for guess in guesses]
        return self.solver, guesses[min(range(len(guesses)), key=scores.__getitem__)], {}, 'profile'

    def reject_solver_solution(self, reason):
        self.get_logger().error(f'GP optimization rejected; no trajectory or control command published ({reason})',
                                throttle_duration_sec=2.0)
        self.replan = True
        self.last_c_opt = np.zeros(self.M)
        self.last_lam_g = np.zeros(self.g.numel())
        self.last_lam_x = np.zeros(self.M)
        evaluation = self.fsup_eval(self.c_numeric, self.p_numeric)
        self.publish_gp_snapshot(evaluation[1:])

    def publish_gp_snapshot(self, lane_state):
        now = default_timer()
        if (not self.publish_gp_state or self.scenario_stamp_ns is None
            or self.gp_state_pub.get_subscription_count() == 0
                or now - self.last_gp_state_time < self.gp_state_period):
            return
        layers = {}
        offset = 13
        for name, count, kernel in (
            ('pedestrian', self.pedestrian_support_count, self.p_ped),
            ('vehicle', self.vehicle_support_count, self.p_veh),
        ):
            alpha = self.p_numeric[offset:offset + count]
            offset += count
            points = self.p_numeric[offset:offset + 3 * count].reshape((count, 3), order='F')
            offset += 3 * count
            layers[name] = {'points': points.tolist(), 'alpha': alpha.tolist(),
                            'kernel': kernel, 'offset': -1.0}
            layers[name].update(self.dynamic_gp_state.get(name, {}))
        for name, points, alpha, kernel in (
            ('ego_lane', lane_state[0], lane_state[2], self.p_el),
            ('opposite_lane', lane_state[1], lane_state[3], self.p_ol),
        ):
            layers[name] = {'points': points.full().tolist(), 'alpha': alpha.full().flatten().tolist(),
                            'kernel': kernel, 'offset': 0.0}
        road = np.column_stack((self.lanes['ego_lane_x'], self.lanes['ego_lane_y'], self.lanes['ego_lane_z']))
        snapshot = {'version': 2, 'frame_id': 'base_link',
                    'stamp_ns': self.scenario_stamp_ns,
                    'planning': getattr(self, 'plan_diagnostics', {}),
                    'times': np.asarray(self.t).flatten().tolist(),
                    'road': road.tolist(), 'road_bounds': self.road_bounds, 'layers': layers}
        try:
            encoded = json.dumps(snapshot, allow_nan=False)
        except (ValueError, TypeError) as error:
            self.get_logger().error(f'GP snapshot contains invalid numeric data; not published ({error})',
                                    throttle_duration_sec=2.0)
            return
        self.gp_state_pub.publish(String(data=encoded))
        self.last_gp_state_time = now
    


# def main(args=None):
#     rclpy.init(args=args)
#     node = pcm_node()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    node = pcm_node()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()