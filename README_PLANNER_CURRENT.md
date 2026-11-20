# Longitudinal Planner: Current Configuration

This summarizes the Python `pcm_longitudinal` planner with the defaults in
[pcm_longitudinal.launch.py](launch/pcm_longitudinal.launch.py). It describes a
longitudinal optimizer following a prescribed route, not a lateral path planner.
The release contains the Python planner and optional GP cost-map viewer only.

## Install On A New Machine

Use Ubuntu 22.04 (or WSL Ubuntu 22.04) with ROS 2 Humble and Python 3.10.
Place this repository alongside the CRP message packages, Autoware message
packages, and `CooperativeResearchPlatform_Simulator` in the ROS workspace.
The actuator model uses the motor efficiency CSV installed by `simulator` unless
you pass `motor_efficiency_map:=/path/to/map.csv`.

From the workspace root, with ROS, `colcon`, `rosdep`, and `pip` already installed:

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src --rosdistro humble -y
sudo apt install build-essential
python3 -m pip install --user -r src/PredictiveCostMapGenerator/requirements.txt
colcon --log-base log-release build --build-base build-release \
       --install-base install-release --packages-up-to plan_predictive_cost_map
source install-release/setup.bash
```

Use the fresh build/install directories above when upgrading an existing
workspace so old native executables and launch files cannot remain visible.
There is no native planner header, NLopt install, or manually copied cache file
to provision. With JIT enabled, the node generates its CasADi libraries on first
start using `gcc`. The default actuator configuration can take several minutes
to compile. Later launches reuse the
libraries for the same configuration. The cache defaults to
`~/.cache/pcm_longitudinal` (or `$XDG_CACHE_HOME/pcm_longitudinal`); use a writable
location and regenerate locally rather than copying binaries between machines.
With no compiler, or when diagnosing startup, launch with
`planner_jit:=false actuator_jit:=false`. This keeps the same model but can be
slower. No separate cache configuration script is required.

Run the release regressions from the same workspace:

```bash
colcon --log-base log-release test --build-base build-release \
       --install-base install-release --packages-select plan_predictive_cost_map \
       --return-code-on-test-failure
colcon test-result --test-result-base build-release --verbose
```

## Settings

| Setting | Current launch value |
| --- | --- |
| Prediction / underlying vehicle | `dynamic_actuators` / dynamic bicycle |
| Horizon / GP and trajectory samples | $T=5\,\mathrm{s}$ / $N=5$, at $t_i=iT/N$ |
| Replanning / internal controller interval | $0.05\,\mathrm{s}$ / $0.05\,\mathrm{s}$ |
| Published speed-command lookahead | $0.25\,\mathrm{s}$ |
| Published steering | `predicted`, sampled at $0.25\,\mathrm{s}$ |
| Actuator integration | `discrete`, stages of at most $0.005\,\mathrm{s}$ |
| Motor efficiency | `spline`; maximum sampled LUT-fit error $0.05$ |
| Dynamic GP compression | `sparse_gp=true`, `gp_reduction_method=global_mean` |
| Selected supports | Per GP reduction mode; full-mean recovery can retain all supports |
| Object / branch capacity | 15 pedestrians, 15 vehicles; at most 2 branches per object |
| Static target / drivable GP supports | 128 / 128; progress-reference samples: 96 |
| Optimization | Dense SLSQP, exact CasADi derivatives, cached compiled oracles |
| Solver tolerance / GP cost limit | $10^{-3}$ / $0.05$ |

These are launch defaults, not necessarily the parameters of an already running node.

## Control And Vehicle Model

Only four coefficients $c_1,\ldots,c_4$ are optimized. They define a speed
**command**, not the physical speed directly:

$$
v_c(t)=v_0+\sum_{j=1}^{4}c_j\left(\frac{t}{T}\right)^j,
\qquad
v_{\mathrm{ref},k}=v_c\!\left(\min(t_k+0.25,T)\right).
$$

The reference is held over each controller interval. The feedback-only speed
controller requests acceleration, with a special braking request for a nonpositive
reference:

$$
a_{\mathrm{req}}=
\begin{cases}
-9.81,&v_{\mathrm{ref}}\le10^{-6},\\
\operatorname{clip}\!\left(6(v_{\mathrm{ref}}-v),-9.81,9.81\right),&\text{otherwise}.
\end{cases}
$$

There is **no slope/load feedforward** in this controller. Road forces act in the
plant. Actual speed is obtained by forward simulation of vehicle and actuators
(single shooting); the command polynomial may become negative to request a stop.

The vehicle state is $(X,Y,Z,\psi,v,v_y,r)$. Away from standstill, the dynamic
bicycle equations underlying the discretization are

$$
\begin{aligned}
F_{yf}&=C_f\left(\delta-\frac{v_y+l_f r}{\bar v}\right),&
F_{yr}&=-C_r\frac{v_y-l_r r}{\bar v},&\bar v&=\max(v,0.5),\\
\dot v&=\frac{F_{\mathrm{drive}}-F_{\mathrm{res}}}{m}
       +r v_y-\frac{F_{yf}\sin\delta}{m},\\
\dot v_y&=\frac{F_{yf}\cos\delta+F_{yr}}{m}-rv,&
\dot r&=\frac{l_fF_{yf}\cos\delta-l_rF_{yr}}{I_z},\\
\dot X&=v\cos\theta\cos\psi-v_y\sin\psi,&
\dot Y&=v\cos\theta\sin\psi+v_y\cos\psi,\\
\dot Z&=v\sin\theta,&\dot\psi&=r,\\
F_{\mathrm{res}}&=\tfrac12\rho C_d A v^2+b_vv
                  +mgC_{rr}\cos\theta+mg\sin\theta.
\end{aligned}
$$

Here $\theta$ is road pitch, $l_r=L-l_f$, and $F_{\mathrm{drive}}$ includes
shaft drive and any additional braking. Current parameters include
$m=1500\,\mathrm{kg}$, $L=2.7\,\mathrm m$, $l_f=1.2\,\mathrm m$,
$I_z=2500\,\mathrm{kg\,m^2}$ and $C_f=C_r=80000\,\mathrm{N/rad}$.
The lateral/yaw update is semi-implicit. Below $2\,\mathrm{m/s}$ it blends toward
kinematics; the dynamic weight is $w=\xi^2(3-2\xi)$ with
$\xi=\operatorname{clip}((v-0.5)/1.5,0,1)$. Speed is clipped nonnegative.
The reported body accelerations correspond to $a_x=\dot v-rv_y$ and
$a_y=\dot v_y+rv$, evaluated through discrete increments.

Route steering is a fixed pure-pursuit law, not another decision variable:

$$
L_d=\max(2,1.5v),\qquad
\delta=\operatorname{clip}\!\left(
\operatorname{atan2}(2Ly_L,x_L^2+y_L^2),-0.6,0.6\right).
$$

The lookahead point $(x_L,y_L)$ comes from an ordered, arc-length-interpolated XYZ
route. Geometry and steering are held within each controller interval. Progress
uses $\Delta s\approx(v_k+v_{k+1})\Delta t/2$; road pitch comes from the route.

## Powertrain And Energy

The prediction retains motor speed $\omega_m$, shaft twist $\phi$, motor torque
$\tau_m$, filtered acceleration $a_f$, and drawn/recovered electrical energy.
With motor ratio $G$, wheel ratio $G_w$, wheel radius $R$, and shaft torque $\tau_s$:

$$
\begin{aligned}
\dot a_f&=(a_{\mathrm{req}}-a_f)/0.05,\\
\dot\phi&=\omega_m/G-G_wv/R,\\
\tau_s&=k_s\phi+d_s\dot\phi,\\
I_{\mathrm{eq}}\dot\omega_m&=G\tau_m-B_{\mathrm{eq}}\omega_m-\tau_s,\\
F_{\mathrm{shaft}}&=G_w\tau_s/R,
\end{aligned}
\qquad
\begin{aligned}
I_{\mathrm{eq}}&=J_mG+J_w/G,\\
B_{\mathrm{eq}}&=B_mG+B_w/G.
\end{aligned}
$$

The filtered request is converted to target torque, saturated to
$[-120,200]\,\mathrm{Nm}$ and rate-limited to $1000\,\mathrm{Nm/s}$.
Additional braking covers deceleration not provided by shaft force. Current
$G=9$, $G_w=1$, $R=0.32\,\mathrm m$, $k_s=18000\,\mathrm{Nm/rad}$ and
$d_s=120\,\mathrm{Nms/rad}$.

**Discrete approximation:** the coupled $(\omega_m,\phi,v)$ subsystem is propagated
with precomputed matrix exponentials. Stage inputs use an averaged torque ramp;
nonlinear loads, braking and lateral coupling are frozen per stage. The
acceleration filter is integrated analytically, and electrical energy uses
two-point Gauss quadrature. This is not exact integration of the complete nonlinear
simulator. `actuator_integration=detailed` instead uses simulator-like microsteps.

$$
P_m=\tau_m\omega_m,\qquad
P_e=\begin{cases}P_m/\eta_{\mathrm{mot}},&P_m\ge0,\\
                  P_m\eta_{\mathrm{regen}},&P_m<0,
     \end{cases}
\qquad
E_{\mathrm{net}}=\int_0^T P_e\,dt=E_{\mathrm{drawn}}-E_{\mathrm{recovered}}.
$$

**LUT approximation:** separate motoring/regeneration $C^2$ cubic-spline surfaces
are fitted to the simulator's bilinear efficiency LUT, then represented in a
separable low-rank form $\eta(\omega,\tau)\approx\sum_{j=1}^{r}f_j(\omega)g_j(\tau)$.
Rank is selected automatically. Startup requires sampled maximum absolute
efficiency error $\le0.05$ on a $241\times201$ validation grid, with fitted values
inside $(0.05,1]$ there. Queries use absolute speed/torque and are clamped to map
axes. The grid check is not an interval certificate. `actuator_efficiency_model=lut`
retains the bilinear LUT; `polynomial` is another optional surrogate. No battery
SOC, voltage dynamics, thermal dynamics, or auxiliary electrical loads are modeled.

## GP Cost Map And Compression

Dynamic layers use $q=(x,y,t)$ and an anisotropic squared-exponential kernel:

$$
k(q,z)=\sigma_f^2\exp\left[-\frac12\sum_d
             \left(\frac{q_d-z_d}{\ell_d}\right)^2\right],\qquad
\alpha=(K+\operatorname{diag}(\nu))^{-1}y,\qquad m(q)=k(q,Z)\alpha.
$$

For an object's retained prediction branch, $p=p_{\mathrm{exist}}p_{\mathrm{branch}}$,
$y_i=1+p\,c_{\mathrm{object}}$, and
$\nu_i=\nu_0+\gamma t_i+p(1-p)c_{\mathrm{object}}^2$.
Pedestrian/vehicle severities are $3/1$. Their kernel parameters
$(\sigma_f,\ell_x,\ell_y,\ell_t)$ are $(1,3.5,1.5,2)$ and
$(e^{-1},e,2.2,1)$. Object paths are interpolated to the 10 prediction times;
without a path, constant-speed/heading extrapolation is used.

**Current sparse approximation:** first fit the original full mean, select 5
supports per retained branch by kernel-scaled farthest-point sampling, then
project the full mean onto the selected kernel basis $U$:

$$
(K_{UU}+\lambda I)\beta=K_{UZ}\alpha,\qquad
\widetilde m(q)=k(q,U)\beta,\qquad
\epsilon=\sigma_f\sqrt{r^TKr+\epsilon_{\mathrm{round}}}.
$$

Here $\lambda=\sigma_f^2\,10^{-10}$ and $r$ embeds $\alpha-\beta$ in the original
basis. The RKHS inequality gives $|m-\widetilde m|\le\epsilon$ in exact arithmetic;
the implementation adds a floating-point rounding allowance, not a formal numeric
certificate. Dynamic costs are $f_p=\widetilde m_p+\epsilon_p-1$ and
$f_v=\widetilde m_v+\epsilon_v-1$. If compression makes even empty space infeasible,
the original full mean is restored with zero approximation error.

Static target/drivable layers $f_t,f_d$ are separate XY GPs with costs $-1/-0.1$;
the drivable surface is an unordered merged waypoint set. They are fitted outside
optimization and are not changed by dynamic mean compression. The combined cost is

$$
S(a,b)=\tfrac12\left(a+b+\sqrt{(a-b)^2+0.01}\right),\qquad
C(q)=S\!\left(S(f_p,f_v),\min(f_t,f_d)\right).
$$

Actuator-mode queries are transformed into the stamped scenario frame, using
prediction time plus scene age. This aligns frames/times but does not eliminate latency.

**Optional, not currently selected:** `gp_reduction_method=global_mean` pools
supports across branches within each dynamic GP. It starts with
$\max(10,\lceil0.2n\rceil)$ candidates, capped by availability, then adds supports
until kernel-scaled coverage radius $0.75$ and mean-error bound $0.02$ are met.
Nearby points may still be added for accuracy; full-mean recovery is retained.
`sparse_gp=false` disables both reductions. Static and dynamic kernels are never
blindly merged. Existing object/branch capacity limits can omit excess input;
compression does not add another actor-removal rule.

## Objective, Constraints And Solver

Writing $C_i=C(q_i)$ and $v_\star$ for the scenario's target speed, current weights give

$$
\begin{aligned}
\min_{c\in\mathbb R^4}J(c)={}&
\frac{0.9}{N}\sum_i(C_i+1)^2
+\frac{0.5}{N\max(v_\star^2,1)}\sum_i(v_i-v_\star)^2\\
&+\frac{0.5}{N\,3^2}\sum_i a_{y,i}^2
+\sum_i\left(e^{5\operatorname{sp}(C_i)}-1\right)
+0.05\frac{E_{\mathrm{net}}}{100000},\\
\operatorname{sp}(u)={}&\frac{\log(1+e^{10u})}{10}.
\end{aligned}
$$

The longitudinal-acceleration objective weight is zero. Energy is **signed net
electrical energy**, not absolute consumption or an instantaneous-power penalty.
At the 10 prediction samples:

$$
|a_x|\le5\,\mathrm{m/s^2},\qquad |a_y|\le3\,\mathrm{m/s^2},\qquad
0\le v\le\max(v_0,v_\star),\qquad C_i\le0.05.
$$

Actuator mode also bounds the held speed-command samples above by
$\max(v_0,v_\star)$, and constrains final progress to the available reference.
Controller/torque/steering saturation is inside the rollout. These sampled GP and
acceleration constraints are **not continuous-time collision guarantees**.

`actuator_solver=sqp` uses SciPy's dense **SLSQP**, with shared CasADi objective/
constraint derivatives and a 30-iteration limit, not CasADi's QP iterations as the
actual optimizer. IPOPT is the same-NLP retry backend (100 iterations, 0.5 s CPU
limit); limited-memory Hessians apply to IPOPT and derivative-oracle setup, not an
exact second-order SLSQP Hessian. Solver status, finiteness, and constraint residuals
are checked before publication; no generated braking-trajectory fallback is used.
Compiled NLP/reporting libraries are cached; first-time compilation can be slow.

**Fixed slots:** with the default model and launch settings, each dynamic GP
layer reserves $15\times2\times5=150$ symbolic
supports for full-mean recovery, even when only a few have nonzero weights.
Each static GP reserves 128 supports. Selected supports are parameters, not extra
decision variables. Fewer active supports do not shrink these compiled expressions;
reduction can add fitting work and alter iteration counts. Multiple compiled
capacity variants are not implemented. A requested 20 Hz period is not a guaranteed
deadline or a claim of closed-loop safety.

## Run

```bash
source ~/ros2_ws/install-release/setup.bash
ros2 launch plan_predictive_cost_map pcm_longitudinal.launch.py
```

Inputs: `/ego`, `/scenario`, and `/simulator/actuator_state`. Outputs include
`/control/command/control_cmd`, `/plan/trajectory`, GP snapshots, and timing JSON on
`/plan/pcm_longitudinal/timing`. To start the optional viewer, pass
`start_costmap_viewer:=true`. The launch uses a separate scenario-preprocessing
worker. Model, reduction, steering, and compilation
settings require restart; topic data remain runtime inputs.

Implementation: [vehicle](plan_predictive_cost_map/planner/dynamics.py),
[rollout](plan_predictive_cost_map/pcm_longitudinal_dynamics.py),
[powertrain/efficiency](plan_predictive_cost_map/pcm_powertrain.py),
[GP reduction](plan_predictive_cost_map/pcm_kernels.py),
[objective/constraints](plan_predictive_cost_map/pcm_longitudinal.py).