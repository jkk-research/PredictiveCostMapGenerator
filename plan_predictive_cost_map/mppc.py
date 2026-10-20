#!/usr/bin/env python3

from timeit import default_timer

import casadi as ca
import numpy as np
from scipy import interpolate
from scipy.ndimage import uniform_filter1d
from geometry_msgs.msg import Vector3

import rclpy
from rclpy.node import Node
from autoware_control_msgs.msg import Control
from crp_msgs.msg import Scenario
from autoware_planning_msgs.msg import TrajectoryPoint
from autoware_planning_msgs.msg import Trajectory
from crp_msgs.msg import Ego
from dataclasses import dataclass, field
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
    # A 2D NumPy array of shape (N, 2) representing [[x1, y1], [x2, y2], ...]
    predicted_path: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

@dataclass
class EgoState:
    vx: float
    ay: float
    ax: float
    yawrate: float
    steeringAngle: float

class mppc_node(Node):
    def __init__(self) -> None:
        super().__init__("mppc")
        self.lanes = {"ego_lane_x": [], "ego_lane_y": [], "left_lane_x": [], "left_lane_y": []}
        self.vehicles = []
        self.pedestrians = []

        self.time_history = []
        self.c_el_history = []

        # 1. Initialize Horizon and Symbols
        self.dt = 0.1
        self.tau_h = 2.0
        self.N = int(self.tau_h/self.dt)
        self.t = ca.vertcat(*[self.dt * (i+1) for i in range(self.N)])        
        self.M = 3  # longitudinal velocity coefficients

        self.ego_state = EgoState(13.0,0,0,0,0)

        self.c_el = [0, 0, 0, 0]
        self.c_ol = [0, 0, 0, 0]

        self.ctrl_cmd = Control()
        self.trajectory = Trajectory()

        self.runningTime = 0.0
        self.plot_results = False
        self.max_pedestrians = 2
        self.max_vehicles = 2
        self.ay_max = 3
        self.ax_max = 5

        self.replan = True
        self.replan_timer = 0.0
        self.c_numeric = np.zeros(self.M)
        self.c_vel = ca.vertcat(self.ego_state.vx, self.c_numeric)

        # symbolic definitions and Casadi optimization setup 
        self.c_sym = ca.MX.sym('c', self.M)              

        self.z_pedestrians_sym = ca.MX.sym('z_pedestrians_sym', self.max_pedestrians*self.N, 3)
        self.z_vehicles_sym = ca.MX.sym('z_vehicles_sym', self.max_vehicles*self.N, 3)
        self.p_sym = ca.vertcat(ca.MX.sym('v_0'), ca.MX.sym('c_ego', 4, 1), ca.MX.sym('c_el', 4, 1), ca.MX.sym('c_ol', 4, 1), ca.reshape(self.z_pedestrians_sym, -1, 1), ca.reshape(self.z_vehicles_sym, -1, 1))

        c_vel = ca.vertcat(self.p_sym[0], self.c_sym)  
        c_ego = self.p_sym[1:5]      

        # Constraints (Kinematics)
        # acceleration
        g_ax = self.f_ego_ax(self.t, c_vel)
        g_ay = self.f_ego_ay(c_vel, c_ego, self.f_ego_x(self.t, c_vel, c_ego), self.t)
        g_vx = self.f_ego_vx(self.t, c_vel)

        g_ax_upper = g_ax - self.ax_max
        g_ax_lower = - self.ax_max - g_ax

        g_ay_upper = g_ay - self.ay_max
        g_ay_lower = - self.ay_max - g_ay

        g_vx_lower = -g_vx

        # Safety cost constraint
        _, fsup_vec = self.cost_sup(self.c_sym, self.p_sym)

        # Stack all nonlinear constraints
        self.g = g_vx_lower
        # ca.vertcat(
        #     g_ax_upper,
        #     g_ax_lower,
        #     g_ay_upper,
        #     g_ay_lower,
        #     g_vx_lower
        # )

        # 1. Get the individual cost terms (Symbolic)
        cost_safety , _= self.cost_sup(self.c_sym, self.p_sym)   # Range [0, 1]
        cost_velocity = self.get_cost_vel(self.c_sym, self.p_sym) # Range [0, ~1]
        cost_ay = ca.sumsqr(self.f_ego_ay(c_vel, c_ego, self.f_ego_x(self.t, c_vel, c_ego), self.t))/(self.ay_max * self.N)
        cost_ax = ca.sumsqr(self.f_ego_ax(self.t, c_vel))/(self.ax_max * self.N)
        # collision penalty only
        cost_crash = ca.sum1(
            ca.exp(5*self.softplus(1/fsup_vec))-1
        )

        # 2. Define Weights
        # weight_safety: Keep the car away from objects/lane bounds
        # weight_vel: Keep the car at the target speed
        w_safety = 0.5
        w_vel = 0.5 
        w_ay = 0.25
        w_ax = 0.25
        w_crash = 50.0
        
        # 3. Combine into final Objective
        # We also keep the regularization for steering smoothness (c_lat)
        obj = (w_safety * cost_safety) + (w_vel * cost_velocity) + (w_crash * cost_crash) + (w_ay * cost_ay) + (w_ax * cost_ax)

        # 6. Setup and Solve NLP
        nlp = {'x': self.c_sym, 'f': obj, 'g': self.g, 'p': self.p_sym}
        opts = {
            'ipopt.print_level':0,
            'print_time':0,
            'ipopt.tol':1e-3,
            'ipopt.warm_start_init_point':'yes',
        }

        self.nlp = nlp
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        x_traj = self.f_ego_x(self.t, c_vel, c_ego)
        y_traj = self.f_ego_y(x_traj, c_ego)
        vx_traj = self.f_ego_vx(self.t, c_vel)

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
        _, fsup_vec = self.cost_sup(self.c_sym, self.p_sym)

        self.fsup_eval = ca.Function(
            'fsup_eval',
            [self.c_sym, self.p_sym],
            [fsup_vec]
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

    def softplus(self, x, beta=10):
        return ca.log(1 + ca.exp(beta*x))/beta

    def ego_callback(self, input_msg):
        self.ego_state = EgoState(input_msg.twist.twist.linear.x, input_msg.accel.accel.linear.x, input_msg.accel.accel.linear.y, input_msg.twist.twist.angular.z, input_msg.tire_angle_front)

    def scenario_callback(self, input_msg):
        # this callback maps the input trajectory to the internal interface
        if hasattr(self, 'vehicles'):
            self.vehicles.clear()
        if hasattr(self, 'pedestrians'):
            self.pedestrians.clear()

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
                if type == 7:
                    # Extract path ONLY for pedestrians
                    path_points = []
                    if obj.kinematics.predicted_paths:
                        first_path = obj.kinematics.predicted_paths[0]
                        for path_pose in first_path.path:
                            path_points.append((path_pose.position.x, path_pose.position.y))
                    
                    # Explicitly pass the parsed path
                    self.pedestrians.append(
                        ObjectState(
                            x=x, 
                            y=y, 
                            theta=theta, 
                            vx=vx, 
                            ax=ax, 
                            type=type, 
                            predicted_path=np.array(path_points) if path_points else np.empty((0, 2))
                        )
                    )
                else:
                    path_points = []
                    if obj.kinematics.predicted_paths:
                        first_path = obj.kinematics.predicted_paths[0]
                        for path_pose in first_path.path:
                            path_points.append((path_pose.position.x, path_pose.position.y))
                    
                    # Explicitly pass the parsed path
                    self.vehicles.append(
                        ObjectState(
                            x=x, 
                            y=y, 
                            theta=theta, 
                            vx=vx, 
                            ax=ax, 
                            type=type, 
                            predicted_path=np.array(path_points) if path_points else np.empty((0, 2))
                        )
                    )

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
    
    def y_curve(self, t, c):
        # t: scalar or vector, c: 4-element vector of coefficients
        return c[3] + c[2]*t + c[1]*t**2 + c[0]*t**3
    
    def f_road_curvature(self, x, c_ego):
        # First derivative
        dy_dx = (
            3*c_ego[0]*x**2 +
            2*c_ego[1]*x +
            c_ego[2]
        )

        # Second derivative
        d2y_dx2 = (
            6*c_ego[0]*x +
            2*c_ego[1]
        )

        # Geometric curvature
        return d2y_dx2 / (1 + dy_dx**2)**1.5

    def f_ego_vx(self, t, c_vel):
        return c_vel[0] + c_vel[1]*t + c_vel[2]*t**2 + c_vel[3]*t**3

    def f_ego_ax(self, t, c_vel):
        return c_vel[1] + 2*c_vel[2]*t + 3*c_vel[3]*t**2

    def f_ego_ay(self, c_vel, c_ego, x, t):
        vx = self.f_ego_vx(t, c_vel)
        
        curvature = self.f_road_curvature(x, c_ego)

        return vx**2 * curvature

    def f_l_y(self, x, c):
        return self.y_curve(x, c)
    
    # Add this new function inside your pcm_node class
    def solve_for_x(self, c, s_vec):

        x = ca.MX(0)
        xs = []

        s_prev = 0

        for i in range(s_vec.numel()):

            s = s_vec[i]

            # distance increment
            ds = s - s_prev

            # polynomial derivative dy/dx
            dy_dx = (
                c[2]
                + 2*c[1]*x
                + 3*c[0]*x**2
            )

            # heading
            theta = ca.atan(dy_dx)

            # integrate x
            dx = ds * ca.cos(theta)

            x = x + dx

            xs.append(x)

            s_prev = s

        return ca.vertcat(*xs)
 
    def f_ego_theta(self, x, c_ego):
        return c_ego[1] + 2*c_ego[2]*x + 3*c_ego[3]*x**2 

    def f_ego_x(self, t, c_vel, c_ego):
        dt = ca.vertcat(t[0], t[1:] - t[:-1])

        vx = self.f_ego_vx(t, c_vel)       
        ds = vx * dt
        s = ca.cumsum(ds)

        return self.solve_for_x(c_ego, s)

    def f_ego_y(self, x, c_ego):
        return self.y_curve(x, c_ego)
        
    def f_obj_x(self, t, vx, x0):
        return x0 + vx * t

    def f_obj_y(self, t, vx, x0, c):
        x = self.f_obj_x(t, vx, x0)
        return self.f_l_y(x, c)

    def cost_sup(self, c, p):

        # This function computes the safety cost based on the current trajectory and the positions of pedestrians, vehicles, and lane boundaries.
        # For this, the distance to the objects and lanes is evaluated, and a cost is computed based on these distances.

        # get the parameters from the concatenated vector
        vx0 = p[0]
        c_ego = p[1:5]
        c_el = p[5:9]
        c_ol = p[9:13]

        # Object and pedestrian location reconstruction
        z_pedestrians_flat = p[13:13+3*self.max_pedestrians*self.N]
        z_vehicles_flat = p[13+3*self.max_pedestrians*self.N:
                            13+3*self.max_pedestrians*self.N + 3*self.max_vehicles*self.N]       
        Z_ped = ca.reshape(
            z_pedestrians_flat,
            self.max_pedestrians * self.N,
            3
        )
        Z_veh = ca.reshape(
            z_vehicles_flat,
            self.max_vehicles * self.N,
            3
        )

        # Reconstruct the full 4-coefficient velocity vector
        c_vel = ca.vertcat(
            vx0,
            c
        )

        # 1. Generate Ego Trajectory (Shape: N x 3)
        x_ego = self.f_ego_x(self.t, c_vel, c_ego)
        y_ego = self.f_ego_y(self.f_ego_x(self.t, c_vel, c_ego), c_ego)
        # We must transpose to get [N x 3] so it matches the Kernel 'X' logic
        ego_traj = ca.horzcat(x_ego, y_ego, self.t)

        # 2. Generate Inducing Points for Lanes (Inducing points along ego x-path)
        # These create the "Safe/Unsafe" zones for the GP        
        ego_lane_pts = ca.horzcat(x_ego, self.f_l_y(x_ego, c_el), self.t)
        opp_lane_pts = ca.horzcat(x_ego, self.f_l_y(x_ego, c_ol), self.t)

        # 3. Compute distances to objects
        # 3.1 distance to pedestrians
        # Reshape pedestrian points to (N, 3, max_pedestrians)

        # Compute distance for each pedestrian at each time step
        # Z_ped has shape (max_pedestrians * N, 3)
        dist_ped = []

        for i in range(self.max_pedestrians):
            # Extract N points for pedestrian i
            ped_i = Z_ped[i * self.N:(i + 1) * self.N, :]

            # L2 distance in (x, y, t), row-wise
            diff = ego_traj - ped_i
            eps = 1e-8
            dist_i = ca.sqrt(ca.sum2(diff**2) + eps)

            dist_ped.append(dist_i)

        # Minimum distance to any pedestrian at each ego trajectory point
        d_ped = dist_ped[0]
        for i in range(1, self.max_pedestrians):
            d_ped = ca.fmin(d_ped, dist_ped[i])

        # 4. Evaluate f_sup
        # This returns an [N x 1] vector of costs (one for each time step)
        fsup_vec = d_ped

        # 4. Convert vector to scalar for optimization
        # Option A: Average cost (smoother gradient)
        # Option B: ca.sum1(ca.fmax(0, fsup_vec)) (penalize only high cost)
        total_cost = self.N / ca.sum1(fsup_vec)**2

        self.debug_d_ped = ca.Function(
            "debug_d_ped",
            [c, p],
            [total_cost]
        )
        
        return total_cost, fsup_vec


    def get_cost_vel(self, c_sym, p_sym):
        # self.target_vel should be defined in __init__, e.g., self.target_vel = 20.0
        target_v = 13.0 
        vx0 = p_sym[0]
        
        # Reconstruct the full 4-coefficient velocity vector
        # [v0, c_opt1, c_opt2, c_opt3]
        c_vel = ca.vertcat(vx0, c_sym)
        
        # Predict vx over the horizon (returns N x 1)
        vx_profile = self.f_ego_vx(self.t, c_vel)
        
        # Calculate error
        vel_error = target_v - vx_profile
        
        # MATLAB: (err' * err) / ( (tau_h/dt) * target_vel^2 )
        # ca.sumsqr(x) is equivalent to x' * x
        denominator = self.N * (target_v**2)
        
        return ca.sumsqr(vel_error) / denominator


    def callback(self):
        t1 = default_timer()  
        t_callback_start = time.perf_counter() 

        now = t1
        self.last_exec = getattr(self, 'last_exec', now)

        nownow = time.perf_counter()

        if hasattr(self, "_last_callback_start"):
            dt = nownow - self._last_callback_start
            self.get_logger().info(f"Callback interval: {dt:.4f}s")

        self._last_callback_start = nownow

        if now - self.last_exec < self.timer_period:
            return

        self.last_exec = now

        # Initial Guess (Warm start)
        if not hasattr(self, 'last_c_opt'): self.last_c_opt = np.zeros(self.M)    
    
        # Ensure we have lane data before proceeding
        if not hasattr(self, 'c_el') or np.all(np.array(self.c_el) == 0):
            self.get_logger().warn("Waiting for lane data...")
            return
        
        # Filling up the objects with numeric values from inputs        
        p_obj = np.ones((self.max_pedestrians * self.N, 3)) * 1000

        if hasattr(self, 'pedestrians'):
            for i, obj in enumerate(self.pedestrians[:self.max_pedestrians]):
                x_obj = obj.predicted_path[:self.N, 0]
                y_obj = obj.predicted_path[:self.N, 1]

                Z_i = np.column_stack([
                    np.array(x_obj).flatten(),
                    np.array(y_obj).flatten(),
                    np.array(self.t).flatten()
                ])

                start = i * self.N
                end = (i + 1) * self.N

                p_obj[start:end, :] = Z_i

        # Filling up the objects with numeric values from inputs        
        p_veh = np.ones((self.max_vehicles * self.N, 3)) * 1000

        if hasattr(self, 'vehicles'):
            for i, obj in enumerate(self.vehicles[:self.max_vehicles]):
                x_obj = obj.predicted_path[:self.N, 0]
                y_obj = obj.predicted_path[:self.N, 1]

                Z_i = np.column_stack([
                    np.array(x_obj).flatten(),
                    np.array(y_obj).flatten(),
                    np.array(self.t).flatten()
                ])

                start = i * self.N
                end = (i + 1) * self.N

                p_veh[start:end, :] = Z_i

        c_ego = self.c_el # ego predicted path from external component (in this case, follow ego lane)

        self.p_numeric = np.concatenate([
            np.array([self.ego_state.vx]), # 0
            c_ego, # 1-4
            self.c_el, # 5-8
            self.c_ol, # 9-12
            p_obj.reshape(-1, order='F'), #13:N*max_ped*3
            p_veh.reshape(-1, order='F') #13+N*max_ped*3:13+N*max_ped*3+N*max_veh*3
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

            d_ped_val = self.debug_d_ped(self.last_c_opt, self.p_numeric).full().flatten()
            self.get_logger().info(f"d_ped: {d_ped_val}")

            sol = self.solver(x0=self.last_c_opt, lam_x0=self.last_lam_x, lam_g0=self.last_lam_g, lbx=lbc, ubx=ubc, lbg=lbg, ubg=ubg, p=self.p_numeric)
            self.c_numeric = sol['x']
            self.last_c_opt = self.c_numeric.full().flatten()
            self.last_lam_x = sol['lam_x'].full()
            self.last_lam_g = sol['lam_g'].full()

            stats = self.solver.stats()

            self.get_logger().info(
                f"status: {stats['return_status']}, "
                f"iterations: {stats['iter_count']}"
            )

            self.replan = False
            self.replan_timer += self.timer_period               
            self.c_vel = ca.vertcat(self.ego_state.vx, self.c_numeric)

        # Evaluate f_sup_vec (= g_cost) for the optimized solution
        f_sup_vec_numeric = self.fsup_eval(self.c_numeric, self.p_numeric)

        # Convert CasADi DM -> NumPy array for printing
        f_sup_vec_numeric = np.array(f_sup_vec_numeric.full()).flatten()

        if np.any(f_sup_vec_numeric > 0.0) or self.replan_timer > 1.0: 
            self.replan = True
            self.replan_timer = 0.0
        else:
            self.replan = False
            self.replan_timer += self.timer_period
            self.get_logger().info(
                f"Maximum value: {np.max(f_sup_vec_numeric)}"
            )

        self.get_logger().info(f"f_sup_vec: {f_sup_vec_numeric}")

        velocityTarget = max(
            self.f_ego_vx(self.timer_period, self.c_vel),
            0
        )

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
        res = self.traj_eval(self.c_numeric, self.p_numeric)

        x_final = res[0].full().flatten()
        y_final = res[1].full().flatten()
        vx_final = res[2].full().flatten()


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

            # velocity      
            pt.longitudinal_velocity_mps = float(vx_final[i])

            # append point
            trajectory_msg.points.append(pt)

        # publish
        self.trajectory_pub.publish(trajectory_msg)
        
        t_end = default_timer()
        self.get_logger().info(f"Loop time: {t_end-t1:.4f}s")

        self.get_logger().info(
            f"Full callback duration: "
            f"{time.perf_counter() - t_callback_start:.4f}s"
        )
    


# def main(args=None):
#     rclpy.init(args=args)
#     node = pcm_node()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    node = mppc_node()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()