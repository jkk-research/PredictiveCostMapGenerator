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

plt.ion() # interactive mode

@dataclass
class ObjectState:
    x: float
    y: float
    theta: float
    vx: float
    ax: float

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

        self.p_ov = [math.exp(0.0), math.exp(1.5), math.exp(1.0), math.exp(0.1)]
        self.p_el = [math.exp(-1.0), math.exp(0.0), math.exp(0.2), math.exp(-100)]
        self.p_ol = [math.exp(-1.0), math.exp(1.0), math.exp(-0.1), math.exp(-3)]


        self.ctrl_cmd = Control()
        self.trajectory = Trajectory()

        self.runningTime = 0.0
        self.plot_results = True
        self.max_objs = 10

        # symbolic definitions and Casadi optimization setup 
        c_sym = ca.MX.sym('c', self.N + self.M)      
        c_lat = c_sym[:self.N]
        c_vel = ca.vertcat(self.ego_state.vx, c_sym[self.N:])

        z_objs_sym = ca.MX.sym('z_objs', self.max_objs*self.N, 3)
        z_objs_flat = ca.reshape(z_objs_sym, -1, 1)
        df_0_sym = ca.MX.sym('df_0')
        v_0_sym = ca.MX.sym('v_0')
        c_el_sym = ca.MX.sym('c_el', 4, 1)
        c_ol_sym = ca.MX.sym('c_ol', 4, 1)

        p_sym = ca.vertcat(z_objs_flat, df_0_sym, v_0_sym, c_el_sym, c_ol_sym)        

    
        

        # declare rosparams
        if not self.has_parameter("/pcm/weights/weight_costmap"):
            self.declare_parameters(
                namespace="",
                parameters=[
                    ("/pcm/weights/weight_costmap", 0.65),
                ],
            )

        # create publishers/subscribers
        self.timer_period = 0.1  # [s]
        self.timer = self.create_timer(self.timer_period, self.callback)

        self.cmd_pub = self.create_publisher(Control, '/control/command/control_cmd', 30)
        self.trajectory_pub = self.create_publisher(Trajectory, '/plan/trajectory', 30)

        self.traj_sub = self.create_subscription(
            Scenario, '/scenario', self.scenario_callback, 10
            )
        
        self.ego_sub = self.create_subscription(
            Ego, '/ego', self.ego_callback, 10
            )
        
        # 1. Initialize Plotting once
        plt.ion() 
        self.fig, (self.ax, self.ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
        self.ax.set_aspect('equal')
        
        # 2. Create "Empty" plot objects that we will update later
        # We use dummy data for now
        self.mesh = self.ax.pcolormesh(np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), 
                                    shading='auto', cmap='hot', vmin=-1.5, vmax=1.5)
        self.cbar = self.fig.colorbar(self.mesh, ax=self.ax)
        self.obj_dots, = self.ax.plot([], [], 'bo', markersize=10, label='Objects')
        
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.legend()

    def ego_callback(self, input_msg):
        self.ego_state = EgoState(input_msg.twist.twist.linear.x, input_msg.accel.accel.linear.x, input_msg.accel.accel.linear.y, input_msg.twist.twist.angular.z, input_msg.tire_angle_front)

    def scenario_callback(self, input_msg):
        # this callback maps the input trajectory to the internal interface
        self.oncomingObjects.clear()
        self.followedObjects.clear()

        for obj in input_msg.local_moving_objects.objects:
            pose = obj.kinematics.initial_pose_with_covariance.pose
            twist = obj.kinematics.initial_twist_with_covariance.twist
            accel = obj.kinematics.initial_acceleration_with_covariance.accel
            x = pose.position.x
            y = pose.position.y

            # Orientation is quaternion — convert to yaw (theta)
            q = pose.orientation
            theta = self.quaternion_to_yaw(q)

            vx = twist.linear.x
            ax = accel.linear.x

            if math.sqrt(x**2+y**2) < 150 and x >= -150:
                if (vx > 5):
                    self.followedObjects.append(ObjectState(x, y, theta, vx, ax))
                elif (vx < -5):
                    self.oncomingObjects.append(ObjectState(x, y, theta, vx, ax))

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
    def f_dyn(self, X, Z, y, p, sigma_n):
        y = ca.reshape(y, -1, 1) # Ensure column vector
        K = self.k(Z, Z, p) + sigma_n**2 * ca.DM.eye(Z.shape[0])
        # K is m x m, y is m x 1 -> solve is m x 1
        # k(X, Z) is n x m -> result is n x 1
        return (self.k(X, Z, p) @ ca.solve(K, y)) - 1.0

    def f_dyn_(self, X, Z, y, p, sigma_n):
        # 1. Force y to be a column vector [m x 1]
        y = ca.reshape(y, -1, 1)
        
        # 2. Compute the Kernel Matrix for observations [m x m]
        K = self.k(Z, Z, p) + sigma_n**2 * ca.DM.eye(Z.shape[0])
        
        # 3. Compute the cross-kernel [n x m]
        k_star = self.k(X, Z, p)
        
        # 4. GP Mean formula: k* @ (K^-1 @ y)
        # Result is [n x 1]
        return (k_star @ ca.solve(K, y)) - 0.5

    # def f_sup(self, X,
    #                 Z_ov, y_ov, p_ov,
    #                 Z_fv, y_fv, p_fv,
    #                 Z_stat, y_stat, p_stat,
    #                 Z_ol, y_ol, p_ol):
    #     avg_dyn = (self.f_dyn(X, Z_ov, y_ov, p_ov, 1) +
    #             self.f_dyn(X, Z_fv, y_fv, p_fv, 1)) / 2
    #     min_dyn = ca.fmin(self.f_dyn_(X, Z_stat, y_stat, p_stat, 1),
    #                     self.f_dyn_(X, Z_ol, y_ol, p_ol, 1))
    #     return ca.fmax(avg_dyn, min_dyn)

    def f_sup(self, X,
                    Z_ov, y_ov, p_ov,
                    Z_stat, y_stat, p_stat,
                    Z_ol, y_ol, p_ol):
        avg_dyn = self.f_dyn(X, Z_ov, y_ov, p_ov, 1)

        min_dyn = ca.fmin(self.f_dyn_(X, Z_stat, y_stat, p_stat, 1),
                        self.f_dyn_(X, Z_ol, y_ol, p_ol, 1))
        return ca.fmax(avg_dyn, min_dyn)

    def y_curve(self, t, c):
        # t: scalar or vector, c: 4-element vector of coefficients
        return c[3] + c[2]*t + c[1]*t**2 + c[0]*t**3

    def f_ego_vx(self, t, c_vel):
        return c_vel[0] + c_vel[1]*t + c_vel[2]*t**2 + c_vel[3]*t**3

    def f_ego_ax(self, t, c_vel):
        return c_vel[1] + 2*c_vel[2]*t + 3*c_vel[3]*t**2

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
        c_vel = ca.vertcat(self.ego_state.vx, c[self.N:])

        # 1. Generate Ego Trajectory (Shape: N x 3)
        x_ego = self.f_ego_x(self.t, c_vel, c_lat, self.ego_state.steeringAngle)
        y_ego = self.f_ego_y(self.t, c_vel, c_lat, self.ego_state.steeringAngle)
        # We must transpose to get [N x 3] so it matches the Kernel 'X' logic
        ego_traj = ca.horzcat(x_ego, y_ego, self.t)

        # 2. Generate Inducing Points for Lanes (Inducing points along ego x-path)
        # These create the "Safe/Unsafe" zones for the GP
        c_el = p[self.max_objs*self.N*3+2:self.max_objs*self.N*3+6]
        c_ol = p[self.max_objs*self.N*3+6:self.max_objs*self.N*3+10]
        ego_lane_pts = ca.horzcat(x_ego, self.f_l_y(x_ego, c_el), self.t)
        opp_lane_pts = ca.horzcat(x_ego, self.f_l_y(x_ego, c_ol), self.t)

        z_objs_flat = p[0:self.max_objs*self.N*3]
        Z_oc = ca.reshape(
            z_objs_flat,
            self.max_objs * self.N,
            3
        )
        y_oc = np.ones((self.max_objs * self.N, 1))*2

        # 3. Evaluate f_sup
        # This returns an [N x 1] vector of costs (one for each time step)
        fsup_vec = self.f_sup(
            ego_traj,
            Z_oc, y_oc, self.p_ov, # Oncoming objects
            ego_lane_pts, ca.DM.ones(self.N, 1) * -1.0, self.p_el, # Ego lane
            opp_lane_pts, ca.DM.ones(self.N, 1) * -0.1, self.p_ol  # Opp lane
        )

        # 4. Convert vector to scalar for optimization
        # Option A: Average cost (smoother gradient)
        # Option B: ca.sum1(ca.fmax(0, fsup_vec)) (penalize only high cost)
        total_cost = ca.sum1(fsup_vec + 1.0) / (2.0 * self.N)
        
        return total_cost, fsup_vec


    def get_cost_vel(self, c_sym):
        # self.target_vel should be defined in __init__, e.g., self.target_vel = 20.0
        target_v = 20.0 
        
        # Reconstruct the full 4-coefficient velocity vector
        # [v0, c_opt1, c_opt2, c_opt3]
        c_vel = ca.vertcat(self.ego_state.vx, c_sym[self.N:])
        
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
        t1 = default_timer()        
       
        # Ensure we have lane data before proceeding
        if not hasattr(self, 'c_el') or np.all(np.array(self.c_el) == 0):
            self.get_logger().warn("Waiting for lane data...")
            return

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


        p_numeric = np.concatenate([
            p_obj.reshape(-1, order='F'),
            np.array([self.ego_state.steeringAngle]),
            np.array([self.ego_state.vx]),
            self.c_el,
            self.c_ol
        ]).flatten()

        # 3. Setup Grid for Visualization
        x_vals = np.arange(0, 150.5, 1.0)  # Coarser step for speed
        y_vals = np.arange(-20, 20.5, 0.25) 
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        points_3d = np.hstack([np.vstack([X_grid.ravel(), Y_grid.ravel()]).T, 
                            np.zeros((X_grid.size, 1))]) 
        X_casadi_grid = ca.DM(points_3d)

        # 4. Define Environment Inducing Points (Lanes)
        x_lane_samples = np.arange(0, 150, 5.0)
        y_el_samples = self.f_l_y(x_lane_samples, self.c_el)
        Z_el = ca.horzcat(x_lane_samples, y_el_samples, np.zeros_like(x_lane_samples))
        y_el_val = ca.DM.ones(x_lane_samples.size, 1) * -1.0

        y_ol_samples = self.f_l_y(x_lane_samples, self.c_ol)
        Z_ol = ca.horzcat(x_lane_samples, y_ol_samples, np.zeros_like(x_lane_samples))
        y_ol_val = ca.DM.ones(x_lane_samples.size, 1) * -0.1

        # Constraints (Kinematics)
        x_ego = self.f_ego_x(self.t, c_vel, c_lat, self.ego_state.steeringAngle)
        y_ego = self.f_ego_y(self.t, c_vel, c_lat, self.ego_state.steeringAngle)
        g = y_ego # We constrain the y-position over the horizon

        # 1. Get the individual cost terms (Symbolic)
        cost_safety, fsup_vec_sym = self.cost_sup(c_sym, p_sym)   # Range [0, 1]
        cost_velocity = self.get_cost_vel(c_sym) # Range [0, ~1]

        # 2. Define Weights
        # weight_safety: Keep the car away from objects/lane bounds
        # weight_vel: Keep the car at the target speed
        w_safety = 0.65
        w_vel = 0.35 
        
        # 3. Combine into final Objective
        # We also keep the regularization for steering smoothness (c_lat)
        obj = (w_safety * cost_safety) + (w_vel * cost_velocity)

        # 6. Setup and Solve NLP
        nlp = {'x': c_sym, 'f': obj, 'g': g, 'p': p_sym}
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.tol': 1e-3, 
            'ipopt.max_iter': 50
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        fsup_eval_fun = ca.Function(
            'fsup_eval_fun',
            [c_sym, p_sym],
            [fsup_vec_sym],
        )

        # Variable Bounds (lbx, ubx)
        lbc = -np.inf * np.ones(self.N + self.M)
        ubc =  np.inf * np.ones(self.N + self.M)
        lbc[:self.N] = -0.3  # Max steering rate [rad/s]
        ubc[:self.N] = 0.3
        
        # Constraint Bounds (lbg, ubg) - Numeric lane tubes
        # We sample the lane center based on a linear x-estimate for numeric bounds
        x_est = np.linspace(0, 100, self.N) 
        lane_center_numeric = self.f_l_y(x_est, self.c_el)
        lbg = lane_center_numeric - 1.5
        ubg = lane_center_numeric + 1.5

        # Initial Guess (Warm start)
        if not hasattr(self, 'last_c_opt'): self.last_c_opt = np.zeros(self.N + self.M)
        
        #sol = solver(x0=self.last_c_opt, lbx=lbc, ubx=ubc, lbg=lbg, ubg=ubg)
        sol = solver(x0=self.last_c_opt, lbx=lbc, ubx=ubc, p=p_numeric)
        c_numeric = sol['x']
        self.last_c_opt = c_numeric.full().flatten()

        steeringAngleTarget = self.f_delta(self.t,c_numeric[:self.N], self.ego_state.steeringAngle)
        velocityTarget = self.f_ego_vx(self.t, ca.vertcat(self.ego_state.vx, c_numeric[self.N:]))
        steeringAngleTarget_next = steeringAngleTarget[0]
        velocityTarget_next = velocityTarget[0]

        steeringAngleTarget_next = min(max(steeringAngleTarget_next, -0.1), 0.1)

        control_cmd_msg = Control()
        control_cmd_msg.stamp = self.get_clock().now().to_msg()
        control_cmd_msg.lateral.steering_tire_angle = float(steeringAngleTarget_next)
        control_cmd_msg.longitudinal.velocity = float(velocityTarget_next)
        self.cmd_pub.publish(control_cmd_msg)

        if self.plot_results:
            # # 7. Evaluate Visuals Numerically
            # # Evaluate Heatmap for the whole grid
            z_objs_flat = p_numeric[0:self.max_objs*self.N*3]
            Z_oc = ca.reshape(
                z_objs_flat,
                self.max_objs * self.N,
                3
            )
            y_oc = np.ones((self.max_objs * self.N, 1))*2
            fsup_grid_vals = self.f_sup(X_casadi_grid, Z_oc, y_oc, self.p_ov, Z_el, y_el_val, self.p_el, Z_ol, y_ol_val, self.p_ol)
            fsup_grid = np.array(fsup_grid_vals.full()).reshape(len(y_vals), len(x_vals))

            # # Evaluate Optimized Path
            traj_eval = ca.Function('traj_eval', [c_sym], [x_ego, y_ego])
            res = traj_eval(c_numeric)
            x_final = res[0].full().flatten()
            y_final = res[1].full().flatten()

            # 8. Plotting
            self.ax.clear()
            im = self.ax.pcolormesh(X_grid, Y_grid, fsup_grid, shading='auto', cmap='gray', vmin=-1.0, vmax=1.0)
            
            if hasattr(self, "cbar"):
                self.cbar.update_normal(im)
                self.cbar.mappable.set_clim(-1.0, 1.0)
                    
            self.ax.set_xlim(0, 150)
            self.ax.set_ylim(-20, 20)
            self.ax.set_title(f"MPC Cost Map | Obj: {sol['f'].full()[0,0]:.3f}")
            self.ax.plot(x_final, y_final, 'r-', linewidth=3, label='Optimized Path')

            current_c_el = self.c_el[-1] 
            self.time_history.append(self.runningTime)
            self.c_el_history.append(current_c_el)
            if len(self.time_history) > 200:
                self.time_history.pop(0)
                self.c_el_history.pop(0)

            self.ax2.clear()
            self.ax2.plot(self.t, steeringAngleTarget, 'b-', linewidth=2)
            self.ax2.set_xlabel("Time [s]")
            self.ax2.set_ylabel("Ego Lane Dist")
            self.ax2.set_title("Ego Lane Parameter Over Time")
            self.ax2.grid(True)


            # Plot objects as dots
            if self.oncomingObjects:
               self.ax.plot([o.x for o in self.oncomingObjects], [o.y for o in self.oncomingObjects], 'bo', label='Objects')

            self.ax.legend(loc='upper right')
            plt.draw()
            plt.pause(0.001)

            self.runningTime = self.runningTime + self.timer_period
        t_end = default_timer()
        self.get_logger().info(f"Loop time: {t_end-t1:.4f}s")
        


def main(args=None):
    rclpy.init(args=args)
    node = pcm_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()