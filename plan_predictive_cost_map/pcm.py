#!/usr/bin/env python3

from timeit import default_timer

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
from dataclasses import dataclass
import math
import matplotlib
matplotlib.use("TkAgg") # GUI backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from rclpy.executors import MultiThreadedExecutor
import threading
plt.ion() # interactive mode

@dataclass
class ObjectState:
    x: float
    y: float
    theta: float
    vx: float
    ax: float
    type: int

@dataclass
class EgoState:
    vx: float
    ay: float
    ax: float
    yawrate: float
    steeringAngle: float

class pcm_node(Node):
    def __init__(self) -> None:
        super().__init__("pcm")
        self.lanes = {"ego_lane_x": [], "ego_lane_y": [], "left_lane_x": [], "left_lane_y": []}
        self.oncomingObjects = [] # empty list
        self.followedObjects = [] # empty list

        self.lock = threading.Lock()

        self.time_history = []
        self.c_el_history = []

        # 1. Initialize Horizon and Symbols
        self.dt = 0.1
        self.tau_h = 2.0
        self.N = int(self.tau_h/self.dt)
        self.t = ca.vertcat(*[self.dt * (i+1) for i in range(self.N)])        
        self.M = 3  # longitudinal velocity coefficients

        self.ego_state = EgoState(0,0,0,0,0)

        self.c_el = [0, 0, 0, 0]
        self.c_ol = [0, 0, 0, 0]

        self.p_ov = [math.exp(0.0), math.exp(1.5), math.exp(1.08), math.exp(1.0)]
        self.p_fv = [math.exp(0.0), math.exp(1.5), math.exp(1.5), math.exp(1.0)]
        self.p_el = [math.exp(-1.0), math.exp(0.0), math.exp(0.2), math.exp(-100)]
        self.p_ol = [math.exp(-1.0), math.exp(1.0), math.exp(-0.1), math.exp(-3)]


        self.ctrl_cmd = Control()
        self.trajectory = Trajectory()

        self.runningTime = 0.0
        self.plot_results = False
        self.max_objs = 2
        self.max_followedObjs = 2
        self.ay_max = 3

        # symbolic definitions and Casadi optimization setup 
        self.c_sym = ca.MX.sym('c', self.N + self.M)              

        self.z_objs_sym = ca.MX.sym('z_objs', self.max_objs*self.N, 3)
        self.z_followedObjs_sym = ca.MX.sym('z_followedObjs', self.max_followedObjs*self.N, 3)
        self.p_sym = ca.vertcat(ca.reshape(self.z_objs_sym, -1, 1),  ca.MX.sym('df_0'), ca.MX.sym('v_0'), ca.MX.sym('c_el', 4, 1), ca.MX.sym('c_ol', 4, 1), ca.MX.sym('alpha_ov', self.max_objs*self.N, 1), ca.MX.sym('alpha_el', self.N, 1), ca.MX.sym('alpha_ol', self.N, 1), ca.reshape(self.z_followedObjs_sym, -1, 1), ca.MX.sym('alpha_fv', self.max_followedObjs*self.N, 1) )

        c_lat = self.c_sym[:self.N]
        c_vel = ca.vertcat(self.p_sym[self.max_objs*self.N*3+1], self.c_sym[self.N:])        

        # Constraints (Kinematics)
        x_ego = self.f_ego_x(self.t, c_vel, c_lat, self.p_sym[self.max_objs*self.N*3])
        y_ego = self.f_ego_y(self.t, c_vel, c_lat, self.p_sym[self.max_objs*self.N*3])
        self.g = y_ego # We constrain the y-position over the horizon

        # 1. Get the individual cost terms (Symbolic)
        cost_safety , _= self.cost_sup(self.c_sym, self.p_sym)   # Range [0, 1]
        cost_velocity = self.get_cost_vel(self.c_sym, self.p_sym) # Range [0, ~1]
        cost_ay = ca.sumsqr(self.f_ego_ay(c_vel, c_lat, self.p_sym[self.max_objs*self.N*3]))/(self.ay_max * self.N)

        # 2. Define Weights
        # weight_safety: Keep the car away from objects/lane bounds
        # weight_vel: Keep the car at the target speed
        w_safety = 0.65
        w_vel = 0.15 
        w_ay = 0.01
        
        # 3. Combine into final Objective
        # We also keep the regularization for steering smoothness (c_lat)
        obj = (w_safety * cost_safety) + (w_vel * cost_velocity) + (w_ay * cost_ay)

        # 6. Setup and Solve NLP
        nlp = {'x': self.c_sym, 'f': obj, 'g': self.g, 'p': self.p_sym}
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.tol': 1e-3, 
            'ipopt.max_iter': 50
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.traj_eval = ca.Function('traj_eval', [self.c_sym, self.p_sym], [x_ego, y_ego])    
        

        # declare rosparams
        if not self.has_parameter("/pcm/weights/weight_costmap"):
            self.declare_parameters(
                namespace="",
                parameters=[
                    ("/pcm/weights/weight_costmap", 0.65),
                ],
            )

        # create publishers/subscribers
        self.timer_period = 0.05  # [s]
        self.timer = self.create_timer(self.timer_period, self.callback)

        self.cmd_pub = self.create_publisher(Control, '/control/command/control_cmd', 30)

        self.trajectory_pub = self.create_publisher(Trajectory, '/plan/trajectory', 30)

        self.traj_sub = self.create_subscription(
            Scenario, '/scenario', self.scenario_callback, 10
            )
        
        self.ego_sub = self.create_subscription(
            Ego, '/ego', self.ego_callback, 10
            )
        
    def ego_callback(self, input_msg):
        with self.lock:
            self.ego_state = EgoState(input_msg.twist.twist.linear.x, input_msg.accel.accel.linear.x, input_msg.accel.accel.linear.y, input_msg.twist.twist.angular.z, input_msg.tire_angle_front)

    def scenario_callback(self, input_msg):
        with self.lock:
            # this callback maps the input trajectory to the internal interface
            self.oncomingObjects.clear()
            self.followedObjects.clear()

            for obj in input_msg.local_moving_objects.objects:
                pose = obj.kinematics.initial_pose_with_covariance.pose
                twist = obj.kinematics.initial_twist_with_covariance.twist
                accel = obj.kinematics.initial_acceleration_with_covariance.accel
                x = pose.position.x
                y = pose.position.y
                type = obj.classification[0].label

                # Orientation is quaternion — convert to yaw (theta)
                q = pose.orientation
                theta = self.quaternion_to_yaw(q)

                vx = twist.linear.x
                ax = accel.linear.x

                if math.sqrt(x**2+y**2) < 150 and x >= -150:
                    if (vx > 5):
                        self.followedObjects.append(ObjectState(x, y, theta, vx, ax, type))
                    elif (vx < -5):
                        self.oncomingObjects.append(ObjectState(x, y, theta, vx, ax, type))

            # lanes
            # Ego lane
            self.lanes["ego_lane_x"].clear()
            self.lanes["ego_lane_y"].clear()
            self.lanes["left_lane_x"].clear()
            self.lanes["left_lane_y"].clear()

            for path_with_rule in input_msg.paths:
                path = path_with_rule.path  # extract the PathWithLaneId

                if not path.points:
                    self.get_logger().warn("Received empty path, skipping")
                    continue

                # Check ego lane
                if abs(path.points[0].point.pose.position.y) < 1.25:
                    for pt in path.points:
                        if pt.point.pose.position.x >= 150:
                            break   # stop looping
                        self.lanes["ego_lane_x"].append(pt.point.pose.position.x)
                        self.lanes["ego_lane_y"].append(pt.point.pose.position.y)

                    self.c_el = np.polyfit(self.lanes["ego_lane_x"], self.lanes["ego_lane_y"], 3)
                else:
                    for pt in path.points:
                        if pt.point.pose.position.x >= 150:
                            break   # stop looping
                        self.lanes["left_lane_x"].append(pt.point.pose.position.x)
                        self.lanes["left_lane_y"].append(pt.point.pose.position.y)
                    self.c_ol = np.polyfit(self.lanes["left_lane_x"], self.lanes["left_lane_y"], 3)


    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)  

    def k(self, X, Z, p):
        """
        X: n x d
        Z: m x d
        p: list or vector [scale, ls1, ls2, ...]
        """
        p = ca.vertcat(*p)
        scale = p[0]
        ls = p[1:].T  # 1 x d
        
        # Broadcast lengthscales to scale coordinates
        X_scaled = X / ca.repmat(ls, X.size1(), 1)
        Z_scaled = Z / ca.repmat(ls, Z.size1(), 1)

        # Compute norms
        X_norms = ca.sum2(X_scaled**2) # n x 1
        Z_norms = ca.sum2(Z_scaled**2) # m x 1
        
        # EXPLICIT BROADCASTING for (X_norms + Z_norms.T)
        # Goal: n x m matrix
        X_norms_expanded = ca.repmat(X_norms, 1, Z.size1()) # n x m
        Z_norms_expanded = ca.repmat(Z_norms.T, X.size1(), 1) # n x m
        
        dist_sq = X_norms_expanded + Z_norms_expanded - 2 * ca.mtimes(X_scaled, Z_scaled.T)
        
        return scale**2 * ca.exp(-0.5 * dist_sq)

    # In f_dyn
    def f_dyn(self, X, Z, y, p, sigma_n, alpha):
        return (self.k(X, Z, p) @ alpha) - 1.0

    def f_dyn_(self, X, Z, y, p, sigma_n, alpha):      
        return (self.k(X, Z, p) @ alpha) - 0.5

    def f_sup(self, X,
                    Z_ov, y_ov, p_ov,
                    Z_ev, y_ev, p_fv,
                    Z_stat, y_stat, p_stat,
                    Z_ol, y_ol, p_ol,
                    alpha_ov, alpha_el, alpha_ol, alpha_ev):
        avg_dyn = (self.f_dyn(X, Z_ev, y_ev, p_fv, 1, alpha_ev) + 
                   self.f_dyn(X, Z_ov, y_ov, p_ov, 1, alpha_ov)) /2

        min_dyn = ca.fmin(self.f_dyn_(X, Z_stat, y_stat, p_stat, 1, alpha_el),
                        self.f_dyn_(X, Z_ol, y_ol, p_ol, 1, alpha_ol))
        return ca.fmax(avg_dyn, min_dyn)

    def y_curve(self, t, c):
        # t: scalar or vector, c: 4-element vector of coefficients
        return c[3] + c[2]*t + c[1]*t**2 + c[0]*t**3

    def f_ego_vx(self, t, c_vel):
        return c_vel[0] + c_vel[1]*t + c_vel[2]*t**2 + c_vel[3]*t**3

    def f_ego_ax(self, t, c_vel):
        return c_vel[1] + 2*c_vel[2]*t + 3*c_vel[3]*t**2

    def f_ego_ay(self, c_vel, dd_f, d_f0):
        delta = self.f_delta(self.t, dd_f, d_f0)
        L = 2.7
        yawRate = ca.tan(delta)/L * self.f_ego_vx(self.t, c_vel)

        return yawRate*self.f_ego_vx(self.t, c_vel)

    def f_l_y(self, x, c):
        return self.y_curve(x, c)

    def f_delta(self, t, dd_f, d_f0):
        # Calculate time steps: [t0, t1-t0, t2-t1, ...]
        t = ca.DM(t) # Forces t to be at least a 1x1 matrix
        if t.numel() == 1:
            dt = t
        else:
            dt = ca.vertcat(t[0], t[1:] - t[:-1])
        
        # change in delta at each step
        delta_change = dd_f * dt
        
        # Integrate: delta_i = d_f0 + sum(dd_f * dt)
        # ca.cumsum computes the running total
        delta = d_f0 + ca.cumsum(delta_change)
        
        return delta

    def f_ego_theta(self, t, c_vel, dd_f, d_f0):
        vx = self.f_ego_vx(t, c_vel)
        delta = self.f_delta(t, dd_f, d_f0)
        t = ca.DM(t) # Forces t to be at least a 1x1 matrix
        if t.numel() == 1:
            dt = t
        else:
            dt = ca.vertcat(t[0], t[1:] - t[:-1])
        
        # Yaw rate approx: (vx / L) * tan(delta)
        L = 2.7
        d_theta = (vx / L) * ca.tan(delta) * dt
        
        # theta_0 is 0 (or pass as argument if ego is already turned)
        theta = ca.cumsum(d_theta)
        return theta

    def f_ego_x(self, t, c_vel, dd_f, d_f0):
        vx = self.f_ego_vx(t, c_vel)
        theta = self.f_ego_theta(t, c_vel, dd_f, d_f0)
        dt = ca.vertcat(t[0], t[1:] - t[:-1])
        
        dx = vx * ca.cos(theta) * dt
        return ca.cumsum(dx)

    def f_ego_y(self, t, c_vel, dd_f, d_f0):
        vx = self.f_ego_vx(t, c_vel)
        theta = self.f_ego_theta(t, c_vel, dd_f, d_f0)
        dt = ca.vertcat(t[0], t[1:] - t[:-1])
        
        dy = vx * ca.sin(theta) * dt
        return ca.cumsum(dy)
        
    def f_obj_x(self, t, vx, x0):
        return x0 + vx * t

    def f_obj_y(self, t, vx, x0, c):
        x = self.f_obj_x(t, vx, x0)
        return self.f_l_y(x, c)

    def cost_sup(self, c, p):
        # c is the optimization variable (MX)
        c_lat = c[:self.N]
        c_vel = ca.vertcat(p[self.max_objs*self.N*3+1], c[self.N:])

        # 1. Generate Ego Trajectory (Shape: N x 3)
        x_ego = self.f_ego_x(self.t, c_vel, c_lat, p[self.max_objs*self.N*3])
        y_ego = self.f_ego_y(self.t, c_vel, c_lat, p[self.max_objs*self.N*3])
        # We must transpose to get [N x 3] so it matches the Kernel 'X' logic
        ego_traj = ca.horzcat(x_ego, y_ego, self.t)

        # 2. Generate Inducing Points for Lanes (Inducing points along ego x-path)
        # These create the "Safe/Unsafe" zones for the GP
        c_el = p[self.max_objs*self.N*3+2:self.max_objs*self.N*3+6]
        c_ol = p[self.max_objs*self.N*3+6:self.max_objs*self.N*3+10]
        ego_lane_pts = ca.horzcat(x_ego, self.f_l_y(x_ego, c_el), self.t)
        opp_lane_pts = ca.horzcat(x_ego, self.f_l_y(x_ego, c_ol), self.t)

        # Generate inducing points for followed objects
        z_followedObjs_flat = p[self.max_objs*self.N*3+10+self.max_objs*self.N+self.N*2:self.max_objs*self.N*3+10+self.max_objs*self.N+self.N*2+self.max_followedObjs*self.N*3]
        Z_fv = ca.reshape(
            z_followedObjs_flat,
            self.max_followedObjs * self.N,
            3
        )
        y_fv = np.ones((self.max_followedObjs * self.N, 1))*1.5

        z_objs_flat = p[0:self.max_objs*self.N*3]
        Z_oc = ca.reshape(
            z_objs_flat,
            self.max_objs * self.N,
            3
        )
        y_oc = np.ones((self.max_objs * self.N, 1))*2

        alpha_oc = p[self.max_objs*self.N*3+10:self.max_objs*self.N*3+10+self.max_objs*self.N]
        alpha_el = p[self.max_objs*self.N*3+10+self.max_objs*self.N:self.max_objs*self.N*3+10+self.max_objs*self.N+self.N]
        alpha_ol = p[self.max_objs*self.N*3+10+self.max_objs*self.N+self.N:self.max_objs*self.N*3+10+self.max_objs*self.N+self.N*2]
        alpha_fv = p[self.max_objs*self.N*3+10+self.max_objs*self.N+self.N*2+self.max_followedObjs*self.N*3:self.max_objs*self.N*3+10+self.max_objs*self.N+self.N*2+self.max_followedObjs*self.N*3+self.max_followedObjs*self.N]

        # 3. Evaluate f_sup
        # This returns an [N x 1] vector of costs (one for each time step)
        fsup_vec = self.f_sup(
            ego_traj,
            Z_oc, y_oc, self.p_ov, # Oncoming objects
            Z_fv, y_fv, self.p_fv, # Followed objects
            ego_lane_pts, ca.DM.ones(self.N, 1) * -1.0, self.p_el, # Ego lane
            opp_lane_pts, ca.DM.ones(self.N, 1) * -0.1, self.p_ol,  # Opp lane
            alpha_oc, alpha_el, alpha_ol, alpha_fv # pre-computed symbolix kernel
        )

        # 4. Convert vector to scalar for optimization
        # Option A: Average cost (smoother gradient)
        # Option B: ca.sum1(ca.fmax(0, fsup_vec)) (penalize only high cost)
        total_cost = ca.sum1(fsup_vec + 1.0) / (2.0 * self.N)
        
        return total_cost, fsup_vec


    def get_cost_vel(self, c_sym, p_sym):
        # self.target_vel should be defined in __init__, e.g., self.target_vel = 20.0
        target_v = 10.0 
        
        # Reconstruct the full 4-coefficient velocity vector
        # [v0, c_opt1, c_opt2, c_opt3]
        c_vel = ca.vertcat(p_sym[self.max_objs*self.N*3+1], c_sym[self.N:])
        
        # Predict vx over the horizon (returns N x 1)
        vx_profile = self.f_ego_vx(self.t, c_vel)
        
        # Calculate error
        vel_error = target_v - vx_profile
        
        # MATLAB: (err' * err) / ( (tau_h/dt) * target_vel^2 )
        # ca.sumsqr(x) is equivalent to x' * x
        horizon_steps = self.tau_h / self.dt
        denominator = horizon_steps * (target_v**2)
        
        return ca.sumsqr(vel_error) / denominator


    def callback(self):
        with self.lock:
            t1 = default_timer()   

            now = t1
            self.last_exec = getattr(self, 'last_exec', now)

            if now - self.last_exec < self.timer_period:
                return

            self.last_exec = now

            # Initial Guess (Warm start)
            if not hasattr(self, 'last_c_opt'): self.last_c_opt = np.zeros(self.N + self.M)    
        
            # Ensure we have lane data before proceeding
            if not hasattr(self, 'c_el') or np.all(np.array(self.c_el) == 0):
                self.get_logger().warn("Waiting for lane data...")
                return
            
            # Filling up the objects with numeric values from inputs        
            p_obj = np.ones((self.max_objs * self.N, 3)) * 1000

            for i, obj in enumerate(self.oncomingObjects[:self.max_objs]):
                x_obj = self.f_obj_x(self.t, obj.vx, obj.x)
                y_obj = self.f_obj_y(self.t, obj.vx, obj.x, self.c_ol)

                Z_i = np.column_stack([
                    np.array(x_obj).flatten(),
                    np.array(y_obj).flatten(),
                    np.array(self.t).flatten()
                ])

                start = i * self.N
                end = (i + 1) * self.N

                p_obj[start:end, :] = Z_i

            K = self.k(ca.DM(p_obj), ca.DM(p_obj), self.p_ov).full()
            K += np.eye(K.shape[0])        

            y_oc = np.ones((self.max_objs * self.N, 1))*2
            alpha_oc = np.linalg.solve(K, y_oc)

            #Followed objects
            p_followedObj = np.ones((self.max_followedObjs * self.N, 3)) * 1000

            for i, obj in enumerate(self.followedObjects[:self.max_followedObjs]):
                x_obj = self.f_obj_x(self.t, obj.vx, obj.x)
                y_obj = self.f_obj_y(self.t, obj.vx, obj.x, self.c_el)

                Z_i = np.column_stack([
                    np.array(x_obj).flatten(),
                    np.array(y_obj).flatten(),
                    np.array(self.t).flatten()
                ])

                start = i * self.N
                end = (i + 1) * self.N

                p_followedObj[start:end, :] = Z_i

            K = self.k(ca.DM(p_followedObj), ca.DM(p_followedObj), self.p_fv).full()
            K += np.eye(K.shape[0])        

            y_fv = np.ones((self.max_followedObjs * self.N, 1))*2
            alpha_fv = np.linalg.solve(K, y_fv)

            # kernel points for ego lane and opponent lane
            if hasattr(self, 'p_numeric'):
                res_estimate = self.traj_eval(self.last_c_opt, self.p_numeric)
            else:
                res_estimate = self.traj_eval(self.last_c_opt, np.zeros(self.p_sym.numel()))
            
            ego_lane_pts = ca.horzcat(res_estimate[0].full().flatten(), self.f_l_y(res_estimate[0].full().flatten(), self.c_el), self.t)
            opp_lane_pts = ca.horzcat(res_estimate[0].full().flatten(), self.f_l_y(res_estimate[0].full().flatten(), self.c_ol), self.t)


            K = self.k(ca.DM(ego_lane_pts), ca.DM(ego_lane_pts), self.p_el).full()
            K += np.eye(K.shape[0])        

            y_el = np.ones((self.N, 1))*(-1)
            alpha_el = np.linalg.solve(K, y_el)

            K = self.k(ca.DM(opp_lane_pts), ca.DM(opp_lane_pts), self.p_ol).full()
            K += np.eye(K.shape[0])        

            y_ol = np.ones((self.N, 1))*(-0.2)
            alpha_ol = np.linalg.solve(K, y_ol)

            self.p_numeric = np.concatenate([
                p_obj.reshape(-1, order='F'),
                np.array([self.ego_state.steeringAngle]),
                np.array([self.ego_state.vx]),
                self.c_el,
                self.c_ol,
                alpha_oc.reshape(-1),
                alpha_el.reshape(-1),
                alpha_ol.reshape(-1),
                p_followedObj.reshape(-1),
                alpha_fv.reshape(-1)
            ]).flatten()

            if hasattr(self, "prev_p"):
                diff = np.linalg.norm(self.p_numeric - self.prev_p)
                self.get_logger().info(f"P diff: {diff:.6f}")

            self.prev_p = self.p_numeric.copy()

            # Variable Bounds (lbx, ubx)
            lbc = -np.inf * np.ones(self.N + self.M)
            ubc =  np.inf * np.ones(self.N + self.M)
            lbc[:self.N] = -0.3  # Max steering rate [rad/s]
            ubc[:self.N] = 0.3        

            sol = self.solver(x0=self.last_c_opt, lbx=lbc, ubx=ubc, p=self.p_numeric)
            c_numeric = sol['x']
            self.last_c_opt = c_numeric.full().flatten()

            steeringAngleTarget = self.f_delta(0.2,c_numeric[2], self.ego_state.steeringAngle)
            velocityTarget = self.f_ego_vx(0.1, ca.vertcat(self.ego_state.vx, c_numeric[self.N:]))
            steeringAngleTarget_next = steeringAngleTarget[0]
            velocityTarget_next = velocityTarget[0]

            steeringAngleTarget_next = min(max(steeringAngleTarget_next, -0.1), 0.1)

            control_cmd_msg = Control()
            control_cmd_msg.stamp = self.get_clock().now().to_msg()
            control_cmd_msg.lateral.steering_tire_angle = float(steeringAngleTarget_next)
            control_cmd_msg.longitudinal.velocity = float(velocityTarget_next)
            self.cmd_pub.publish(control_cmd_msg)

            # getting the trajectory
            res = self.traj_eval(c_numeric, self.p_numeric)
            x_final = res[0].full().flatten()
            y_final = res[1].full().flatten()

            # publishing the trajectory
            trajectory_msg = Trajectory()
            trajectory_msg.header.stamp = self.get_clock().now().to_msg()
            trajectory_msg.header.frame_id = "base_link"

            for i in range(len(x_final)):
                pt = TrajectoryPoint()

                # position
                pt.pose.position.x = float(x_final[i])
                pt.pose.position.y = float(y_final[i])
                pt.pose.position.z = 0.0           

                # append point
                trajectory_msg.points.append(pt)

            # publish
            self.trajectory_pub.publish(trajectory_msg)
            
            t_end = default_timer()
            self.get_logger().info(f"Loop time: {t_end-t1:.4f}s")
        


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