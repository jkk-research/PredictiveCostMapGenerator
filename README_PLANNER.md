# GP trajectory planning, MPC tracking, and safety supervision

```text
scenario + ego -> pcm_planner -> /plan/trajectory
trajectory + ego -> mpc_tracker -> /control/tracking/control_cmd
candidate + scenario + ego + trajectory -> safety_supervisor
                                      -> /control/command/control_cmd
```

`pcm_planner` never publishes actuator commands. `mpc_tracker` never subscribes
to scenario data or evaluates the GP. `safety_supervisor` independently checks
hazards/freshness and is the sole final command publisher. It is not a second
planner and does not choose overtaking maneuvers. Existing `pcm_longitudinal`,
`pcm`, `pcm_combined`, and their old launch behavior are retained; do not run
those command publishers alongside this stack.

## Standalone Longitudinal Planner

To run `pcm_longitudinal` independently of the newer planner/tracker stack:

```bash
colcon build --packages-select plan_predictive_cost_map
source install/setup.bash
ros2 launch plan_predictive_cost_map pcm_longitudinal.launch.py
```

The launch file contains the node's startup parameters. For example,
`lookahead_time:=3.0 prediction_steps:=30` sets a three-second horizon with 30
samples. Target speed comes from `paths[0].traffic_rules.maximum_speed` in each
`/scenario` message, including zero-speed targets. Invalid negative or nonfinite
speeds prevent planning on that input. The prediction timestep is
`lookahead_time / prediction_steps`; `timer_period` independently controls the
planning callback interval. `command_lookahead_time` selects the velocity sample
used for the control command and must lie within the horizon.

Lane GP kernel arrays are `[scale, length_x, length_y]` and use only XY
distances. Dynamic-object arrays remain `[scale, length_x, length_y, length_time]`
and use XY-time distances. All entries are actual positive values, not logarithms.
Lane kernels have no temporal parameter. Dynamic kernels retain their existing
XY-time distance calculation; probability weighting and observation noise are
handled separately when fitting their GP coefficients.
Both prediction models enforce
`abs(predicted_lateral_acceleration) <= max_lateral_acceleration` (default
3 m/s^2) at prediction samples. The objective also includes
`weight_lateral_acceleration * mean((predicted_lateral_acceleration / max_lateral_acceleration)^2)`.
The weight defaults to 0.5 and is configurable in the launch file; restart the
planner after changing it. The weight encourages comfort below the hard limit.
This is a sampled prediction bound, not a guarantee on actual simulator
acceleration between samples or under model/tracking error. Joint GP, speed,
acceleration and reference constraints may still make an input infeasible;
no alternate trajectory is published to bypass these limits.

Trajectory Z is interpolated by progress along the ordered XYZ reference, not
by sorted X. Repeated and decreasing X coordinates are supported.
Vehicle predictions
now include road-grade dynamics; published trajectory heights are projected
onto the scenario road, consistent with the simulator's road-height update. Object
predictions are resampled using their supplied timestep and held at their last
point after they end; missing predictions use constant-velocity extrapolation.
Only this node should publish the final trajectory and control command when
running this launch.

### Target Path and Drivable Waypoint Set

The first `/scenario` path is an ordered XYZ reference. Once per scenario update,
the planner removes consecutive duplicate points, computes cumulative 3D travel
distance, locates the ego's progress by a numeric segment projection, and prepares
`reference_samples` (96 by default) uniformly spaced samples. CasADi uses a
parametric cubic B-spline to evaluate XYZ, heading, grade and curvature at
predicted progress. Interpolation coefficients are computed numerically once
per scenario update, not inside the optimizer. Continuous internal derivatives
avoid the tangent jumps of piecewise-linear interpolation. There is no global `y(x)` fit or
nearest-point search in optimizer evaluations. Both models constrain predicted
progress to the supplied reference length so clamping cannot hide travel beyond
known geometry. This is fixed-reference longitudinal planning, not lateral planning.

Both static GPs are fitted outside the optimizer. `target_support_points` (128
by default) selects a spatially representative subset of actual target waypoints;
the observations are `ego_lane_cost`. Numeric fitting solves
`(K + noise_covariance) * alpha = observations`. The fixed supports, kernel
settings and alpha are passed into CasADi, which only evaluates the means.
The target GP uses XY locations as a set; reference progress preserves their
original ordering. Changes in an optimizer candidate do not trigger a GP refit.
No JIT/cache or symbolic-nearest-waypoint experiment is enabled.

All remaining scenario paths are merged into one unordered XY drivable waypoint
set. They are not fitted as separate lane GPs and no segment is created between
unrelated road waypoints. Thus `[target, merged_drivable_set]` and
`[target, road_part_1, road_part_2, ...]` produce the same planner road GP.
The current simulator still publishes individual corridors because their
topology and boundaries are used for occlusion and road visualization; its
message format is not changed by this planner-side aggregation.

The drivable GP is fitted once per scenario update from actual spatially spread
waypoints, capped by `drivable_support_points` (128 by default), independently of
the ego candidate trajectory. The support selection is order-independent.
The compatibility parameter names `opposite_lane_cost` and `opposite_lane_kernel`
now describe this complete drivable set: default cost -0.4, XY lengthscales
`[exp(1), 2.0]`, and `drivable_noise_std=0.05`. This wider lateral scale covers
the region between nearby road waypoints, rather than rewarding only a thin
centreline. Missing drivable geometry has zero GP contribution, not a synthetic
lane at Y=0. The viewer receives the exact road supports and fitted coefficients
under the existing `opposite_lane` layer key.

RBF means are smooth, not binary road masks: small negative tails can extend
past a road edge. The finite support budget is also an approximation. Velocity
or acceleration command polynomials remain optimization variables, separately
from the progress-interpolated reference geometry. The planner no longer discards road
points at a fixed local X=150 m cutoff; usable geometry follows `/scenario`.

Object RBF evaluation uses direct coordinate differences in planner and viewer
to avoid squared-norm cancellation. Nonfinite solver diagnostics use JSON null,
and invalid GP snapshots are rejected without terminating the planner. Invalid
solver output is still rejected without publishing a substitute trajectory or
control command.

In a bounded highway-entry replay at 35 m/s initial speed, both model modes
ran past map X=400 m with finite snapshots, no reproduced JSON crash, and no
`Invalid_Number_Detected` reports. Median drivable means at retained supports
were approximately -0.397. Planner commands were disconnected for this replay
to force coverage of the merge geometry; this is not closed-loop safety evidence.

### Prediction Model Selection

Select the planner model at startup, independently of the simulator's model:

```bash
ros2 launch plan_predictive_cost_map pcm_longitudinal.launch.py prediction_model:=kinematic
```

```bash
ros2 launch plan_predictive_cost_map pcm_longitudinal.launch.py prediction_model:=dynamic
```

```bash
ros2 launch plan_predictive_cost_map pcm_longitudinal.launch.py prediction_model:=dynamic_light
```

The saved launch currently defaults to `dynamic_light`; the standalone node's
parameter default remains `dynamic`. Restart the planner to change models; the
`prediction_model` parameter is read-only after startup because decision-vector
dimensions and compiled solver functions differ. This does not switch the
simulator's `vehicle_model` parameter.

`kinematic` restores the previous three-coefficient velocity polynomial
`v(t)=v0+c1*t+c2*t^2+c3*t^3`, exact longitudinal distance integration, and
following the progress-interpolated XYZ reference. It does not construct or evaluate the dynamic
integrator, model road loads, or use dynamic actuator constraints. The wheelbase
is only used for kinematic steering metadata; dynamic physical parameters do
not affect its predicted speed. Published road elevation is retained.

`dynamic` uses the four-coefficient requested-acceleration polynomial and
dynamic vehicle model below. `dynamic_light` uses longitudinal load dynamics
and assumes perfect reference-path following, without tire-slip dynamics.
All three modes use the same GP fitting, risk bound,
objective weights, scenario speed limit and target-velocity command topic.
Slower dynamic solves reflect the additional integration and derivative work,
but iteration count, conditioning and infeasibility also affect runtime. Neither
mode selection is itself evidence of safe crossing behavior.

### Longitudinal Dynamic Light

`dynamic_light` uses inverse longitudinal dynamics, without forward integration.
It optimizes four velocity-polynomial coefficients with horizon-normalized time:

```text
tau = t / horizon
v(t) = v0 + b1*tau + b2*tau^2 + b3*tau^3 + b4*tau^4
s(t) = v0*t + horizon*(b1*tau^2/2 + b2*tau^3/3 + b3*tau^4/4 + b4*tau^5/5)
a_net(t) = (b1 + 2*b2*tau + 3*b3*tau^2 + 4*b4*tau^3) / horizon
resistance(v, pitch) = rho*Cd*A*v^2/(2*m) + viscous*v/m
                      + rolling*g*cos(pitch) + g*sin(pitch)
a_request(t) = a_net(t) + resistance(v(t), pitch(s(t)))
```

Speed and progress are evaluated analytically. Road grade comes from the same
reference spline at the candidate's progress; dense load checks evaluate only
its heading and pitch channels, without changing their values or derivatives.
`dynamic_light.constraint_step` (default 0.1 s) controls actuator/speed sampling, not
integration. Boundaries include prediction samples and command time, with an
additional midpoint in every interval (0.05 s effective spacing). It is separate
from `vehicle.control_step`, which only affects the full `dynamic` model.
`vehicle.integration_step` is unused in
this model. `vehicle.use_road_gradient` and `vehicle.road_pitch` configure loads.

Required acceleration and speed remain unclipped in trial evaluations. Explicit
actuator and speed bounds reject invalid candidates at the dense sample times.
The fixed initial speed is not an optimization constraint. Net longitudinal
acceleration, curvature-based lateral acceleration `v^2 * curvature`, GP cost and
available-reference constraints remain active at prediction samples. These are
sampled bounds, not continuous-time guarantees between samples. A global velocity
polynomial can touch zero and reaccelerate, but cannot represent a finite exact
stop-and-wait plateau unless it is identically zero. No braking fallback is added.

XYZ, heading, elevation and curvature come from the ordered progress reference.
There is no integrated lateral velocity, yaw response, tire slip or pure-pursuit
controller. Measured acceleration, lateral velocity, yaw rate and steering do not affect this
prediction. Output remains target velocity at `command_lookahead_time`, not an
acceleration command. This is a reduced-order approximation, not simulator
dynamic-bicycle parity or a prediction of lateral tracking error.

This changes the trajectory family: velocity is now quartic, and requested
acceleration is generally not polynomial because it includes speed- and
position-dependent loads. Zero coefficients mean constant speed with exactly
compensated loads, not continuation of measured acceleration. The existing full
`dynamic` model retains its acceleration polynomial and integrator unchanged.

Tests cover exact velocity/progress polynomials, inverse force balance against
high-accuracy forward integration, automatic map-pitch/elevation handling,
independence from actuator sampling, touching zero speed and reaccelerating,
and rejection of reverse trials between prediction samples. Selected spline
channels are checked against full-geometry values and first/second derivatives.

Frozen 4 s/20-sample problems with exact Hessians measured approximately 122 ms
straight and 194 ms curved cold solves, and 121/145 ms warm solves. A pedestrian
conflict took 2.18 s cold (113 iterations) and 166 ms warm. Dense grade and
constraint evaluation and difficult convergence still dominate some cases;
removing integration has not established real-time or closed-loop safety.
Earlier RK4 dynamic-light timing and simulator results describe the superseded
forward formulation, not this inverse model.

An installed inverse-model replay with the viewer and protected-left simulator
(13 m/s initial speed) produced 200 trajectories and commands in 22 simulated
seconds, reaching map (155.25, 49.05) on the outgoing road. All 173 captured
planning snapshots reported `Solve_Succeeded`, but the maximum wall-clock
command gap was 1.22 s, above the 1 s simulator timeout. This is not a
command-continuity or safety acceptance pass. The user's launch tuning,
including pedestrian lateral kernel lengthscale 2.5 m, was retained.

### Dynamic-Light Real-Time Measures

These change computation and initialization, not the objective, constraints,
GP fitting or acceptance test:

- Reference splines multiply scenario coefficients by a fixed B-spline basis.
  Values and derivatives equal the previous parametric spline (tested against
  SciPy); the exact Hessian is about half as expensive.
- The light rollout is built once and inlined; GP safety, velocity cost and
  constraints reuse it.
- Each cycle first time-shifts the last accepted velocity polynomial exactly to
  the current time. If its maximum constraint value is at most 0.05, IPOPT starts
  there with its multipliers and a small initial barrier (`mu_init` 1e-3).
  Otherwise the planner scores the shifted plan and nine simple cruise, brake
  and accelerate profiles with the actual NLP constraints/objective and starts
  the default solver from the best. These are initial guesses only: a plan is
  published only if IPOPT succeeds and all constraints satisfy the tolerance.
  No alternative trajectory is generated after a rejection.
- A separate process (`scenario_worker`, default true) deserializes `/scenario`
  and prepares the reference and static GPs. The planner applies the newest
  result at the start of each cycle. Out-of-optimizer GP fits use NumPy.
- Timer ticks are accepted after half a period, so jitter does not skip cycles.

Installed live 22 s runs with the viewer, dynamic simulator and 13 m/s initial
speed measured:

| Scenario | Command rate | Median / p95 / max gap | Median / p95 solve | Solver outcome |
| --- | --- | --- | --- | --- |
| Protected left | 17.6 Hz | 51 / 106 / 271 ms | 23 / 128 ms | 288/288 succeeded |
| Roundabout | 14.6 Hz | 54 / 131 / 1009 ms | 58 / 161 ms | 158 succeeded, 2 infeasible |
| Occluded pedestrian | 17.3 Hz | 50 / 75 / 2541 ms | 18 / 210 ms | 127 succeeded, 8 infeasible |

Long gaps coincide with infeasible solves, not slow solves. In the occluded
pedestrian case, the pedestrian appears 18.6 m ahead at 10.9 m/s. The minimum
stopping distance at the 5 m/s^2 limit is about 12 m, so every plan stays
inside the pedestrian GP's 5 m longitudinal lengthscale and exceeds
`max_gp_cost`. Once the pedestrian is ahead, even standing still evaluated to
0.67. Those cycles publish no command, and the simulator timeout brake can act.
This is a property of the current risk model and limits, not a solver failure.
The timings are not a hard real-time guarantee.

The two-pedestrian synthetic stress case was also mostly infeasible, and its
accepted plans exceeded the GP limit between the 20 prediction samples.
The GP limit is only enforced at prediction samples.

### Acceleration Polynomial and Dynamic Vehicle

In dynamic mode, the optimization variables are four coefficients of a cubic requested
drive/brake acceleration, not a velocity polynomial:

```text
a_request(t) = a_base + c0 + c1*t + c2*t^2 + c3*t^3
```

`a_base` is inferred from measured longitudinal acceleration and current model
loads, giving a neutral zero-coefficient warm start. All four coefficients,
including the constant correction `c0`, are free optimization variables. The
requested acceleration is the input to the dynamic bicycle model; its predicted
position, speed and net acceleration feed the GP, speed objective and kinematic
limits. Requested drive/brake limits are checked at every model-control step.
During optimization the model is extended beyond those limits without input
clipping to avoid flat trial-point gradients; accepted plans must satisfy the
explicit actuator bounds. Other callers retain simulator-style clipping.

The model reuses the CasADi counterpart of the simulator's dynamic update:
longitudinal resistance and grade loads, front/rear tire stiffness, lateral
velocity, yaw rate, low-speed blending, and semi-implicit lateral integration.
Measured map pose, road pitch, longitudinal/lateral velocities, yaw rate,
acceleration and steering initialize every prediction. Integration takes place
in map coordinates and GP query positions are transformed back to the current
scenario's `base_link` frame. Path and elevation use the precomputed progress
table. Progress advances by predicted forward travel; steering aims at the
lookahead position in that table. There is no per-integration-step nearest search.
The finite table resolution and reference-following controller can still produce
tracking differences on tight turns.

Planner parameters have their own `vehicle.*` namespace in
`launch/pcm_longitudinal.launch.py`. They are not read from or written to the
simulator node. Initial values match the simulator: wheelbase 2.7 m, mass
1500 kg, yaw inertia 2500 kg m^2, front axle distance 1.2 m, front/rear cornering
stiffness 80000 N/rad, air density 1.225 kg/m^3, drag coefficient 0.3, frontal
area 2.2 m^2, rolling coefficient 0.012, viscous resistance 5 N s/m, acceleration
and braking actuator limits 9.81 m/s^2, and steering limit 0.6 rad. Low-speed
blend thresholds are 0.5 and 2.0 m/s; reference lookahead is 1.5 s with a 2 m
minimum. `vehicle.control_step=0.05` s and `vehicle.integration_step=0.005` s
set the prediction's control and integration intervals. `vehicle.use_road_gradient`
and `vehicle.road_pitch` independently configure planner load modeling.

The external command remains `Control.longitudinal.velocity`, sampled from the
dynamic prediction at `command_lookahead_time`; it does not switch the simulator
to acceleration mode. The simulator's speed controller still determines the
executed acceleration, so identical acceleration-input dynamics do not imply
identical closed-loop velocity tracking. Trajectory points include predicted
velocity, net acceleration, steering, lateral velocity, yaw rate and orientation.

Before the nearest-waypoint rewrite, the 4 s default prediction's simple
flat-road solves measured roughly 0.13--0.18 s on this WSL environment,
exceeding the requested 0.05 s callback
period. These timings are not a 20 Hz real-time guarantee or a stopping-safety
claim. The production-launch crossing acceptance test also still fails: the
dynamic optimizer produced command gaps of approximately 10 s and infeasible
solves in the measured run. Its observed zero speed cannot be credited as a
valid planned stop because the simulator command timeout can intervene.
The reverted literal-waypoint experiment was too slow. With the new fixed
progress table and pre-fitted static GPs, the same traffic-free roundabout
kinematic cases measured about 25--35 ms planning plus 15 ms scenario preparation,
with 0.04--0.07 m reference error. Dynamic cases remained slower (roughly
0.3--2.8 s) and showed steering tracking error. These are isolated measurements,
not real-time or closed-loop safety guarantees; traffic infeasibility remains open.
A live C++ parity test checks identical acceleration, steering and grade
inputs independently of that closed-loop tracking difference:

```bash
PCM_DYNAMIC_MODEL_ROS_TEST=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m pytest test/test_pcm_longitudinal_ros.py::test_acceleration_model_matches_live_cpp_simulator -q
```

### Protected Left Turn

The vehicle GP lateral lengthscale is 2.2 m, rather than the earlier
`exp(1.5)` (4.48 m). With the old width, an oncoming car at a 3.4 m lane
offset could make even a stationary waiting point exceed `max_gp_cost`.
The regression checks adjacent-lane waiting and same-lane conflict separately;
severity, observation noise and the GP limit remain unchanged.

An isolated 30 s protected-left replay with kinematic prediction, initial speed
13 m/s, occlusion, virtual objects and the GP viewer completed the turn with
229 successful solver snapshots, no rejected snapshots and a maximum wall-clock
command gap of 0.48 s (below the simulator's 1 s timeout). Minimum approach
speed was 5.66 m/s. This checks planning continuity for that run, not full
collision safety or dynamic-mode closed-loop performance. No fallback was added.

That continuity result predates enabling the lateral penalty and hard limit.
Two subsequent 22 s kinematic replays with both enabled completed the turn
with measured lateral peaks of 1.66 and 2.15 m/s^2, but had two and one
transient infeasible solves respectively. Maximum command gaps were 2.10 and
1.35 s, exceeding the simulator timeout. One rejection was localized near
map X=140 m on the approach. These runs are not command-continuity passes;
the hard limits remain enforced and rejected solves publish no new command.

### Crossing Calibration

The standalone parameters were investigated against
`occluded_pedestrian_costmap_data.mat` with the simulator's speed-command mode,
internal lateral controller, and 50 km/h initial speed:

| Parameter | Previous | Calibrated |
| --- | --- | --- |
| `lookahead_time` | 2 s | 4 s |
| `prediction_steps` | 20 | 20 |
| `pedestrian_kernel` | `[0.3679, 2.7183, 2.7183, 2.7183]` | `[1.0, 5.0, 1.5, 0.5]` |
| `weight_velocity` | 0.1 | 0.5 |
| `weight_crash` | inactive | 1.0 |

The longer horizon covers the roughly 2.8 s stopping time at 50 km/h under the
planner's 5 m/s^2 deceleration limit. Keeping 20 samples changes prediction dt
from 0.1 to 0.2 s without increasing the GP support budget. The larger kernel
scale strengthens pedestrian influence relative to its observation noise;
the 5 m longitudinal lengthscale makes the approach gradient less localized.
Shorter temporal and narrower lateral lengthscales reduce trajectory smearing.
The higher speed weight helps progress after yielding; it is paired with an
active collision penalty. Raising the speed weight alone caused overlap in a
trial and is not recommended. The pre-existing collision term
`sum(exp(5 * softplus(field)) - 1)` now enters the objective through
`weight_crash`. Setting that weight to zero restores its previous inactive
behavior. Severity and uncertainty formulas are unchanged.

Earlier isolated headless runs passed, but the normal launch with the viewer
also reproduced failure to stop. Those earlier results do not establish safe
crossing behavior. Latest-only scenario and ego queues reduce input backlog;
the acceptance test now runs the actual launch files with the viewer active.

Trajectory selection uses the GP objective, acceleration and nonnegative-speed
constraints, and the original MATLAB bound `f_sup <= 0.05` at every prediction
sample (`max_gp_cost`). Predicted speed cannot exceed the scenario speed limit,
except that an existing overspeed is allowed to be reduced gradually rather
than demanding an instantaneous speed drop. There is no generated braking fallback,
geometric pedestrian-clearance constraint, or probability-threshold collision
mask. An unsuccessful, nonfinite, or constraint-violating solve produces an
error and no new trajectory/control command. The simulator may independently
brake after its command timeout; that must not be counted as GP planning success.

The GP safety term follows the MATLAB mean-of-squares expression
`sum((field + 1)^2) / N`, not `sum(field + 1)^2 / N`. The exponential crash
penalty above is still different from MATLAB's thresholded quadratic crash
term and has not been silently replaced. Safe crossing behavior remains a
validation requirement, not an assumed consequence of removing the fallback.
The earlier velocity-polynomial version used exact longitudinal integration
after a rectangle-rule correction. That predictor has now been replaced by the
acceleration-input dynamic model described above.

The GP-bound production-launch run achieved clearance and zero approach speed,
but it is still a failed acceptance test: infeasible solves caused command gaps
longer than the simulator timeout. Timeout braking is not a GP-generated stop.
Those measurements precede the dynamic-model change. The new model supports
non-reversing stopped states through its physical speed saturation, but this
does not establish feasible, safe closed-loop crossing behavior by itself.

An opt-in regression checks yielding, subsequent progress and command continuity:

```bash
PCM_LONGITUDINAL_CROSSING_TEST=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m pytest test/test_pcm_longitudinal_ros.py::test_default_crossing_yields_and_resumes -q -s
```

Run from this package directory in a sourced, built ROS workspace. The test uses
an isolated ROS domain and approximates the ego footprint as 4.5 by 1.85 m and
the pedestrian radius as 0.3 m. This calibration is not a collision-avoidance
guarantee: the collision penalty remains soft, and the planner still has no
hard pedestrian-clearance constraint. Persistent occlusion hypotheses can
continue to delay progress even after the visible pedestrian passes.

### Virtual Objects and GP Uncertainty

Virtual actors from occlusion are included through the ordinary `/scenario`
moving-object array, using their published existence probability and prediction
branches. They are not regenerated by the planner. For each admitted branch:

```text
E = object.existence_probability
C = predicted_path.confidence
S = pedestrian_cost or vehicle_cost
q = E * C
training_target = 1 + q * S
noise_variance(t) = noise_base + noise_growth * t + q * (1 - q) * S^2
alpha = solve(K + diag(noise_variance), training_target)
dynamic_mean = K_query * alpha - 1
```

The target's `+1` compensates for the dynamic mean's `-1` offset. Existence
probability and branch confidence both weight severity. This restores the
MATLAB script's trajectory-probability weighting for detected objects while
retaining existence weighting for virtual objects. The previous observation
`1 + E * S` incorrectly retained high cost for nearly-zero-confidence branches.
As in that script, Bernoulli variance is largest at joint probability 0.5; it is
not an inverse-confidence noise rule. A low-confidence trajectory is suppressed
by its lower observation, not by noise growing without bound. Actual fitted
means also depend on overlapping kernel supports and observation noise, so they
need not equal `q * S` exactly or vanish near a likely trajectory.

The noise parameters are diagonal covariance entries, not standard deviations.
Defaults from the MATLAB prediction-uncertainty definitions are
`pedestrian_noise_base=0.02`, `pedestrian_noise_growth=0.25`,
`vehicle_noise_base=0.01`, and `vehicle_noise_growth=0.05`. They replace the old
constant `object_noise_std` setting. Time is measured in prediction seconds.
These rules apply to both observed and virtual moving objects. Zero-existence
objects and zero-confidence branches are excluded. Objects without predictions
use the existing constant-velocity fallback, with branch confidence 1.

`max_pedestrians` and `max_vehicles` remain physical-object limits (both 2 by
default), in incoming message order. `max_predicted_paths` defaults to 2 branches
per object, selected by highest confidence without renormalizing their
probabilities. The fixed support count per class is
`max_objects * max_predicted_paths * prediction_steps`. Exceeding either limit
emits a throttled warning; virtual objects beyond the physical-object limit
are still omitted. Unused support slots have zero targets and no cost influence.
Increase these limits deliberately: fitting and visualization work grows with
the support count.

The GP snapshot includes per-support `targets`, `noise_variances`,
`existence_probability`, and `trajectory_confidence` for inspection. The viewer
uses the exact fitted coefficients, so no separate uncertainty tuning or GP
refitting is performed there. The lane GPs and cost-combination rules are
unchanged.

### Independent GP Cost Surface

```bash
ros2 launch plan_predictive_cost_map pcm_longitudinal.launch.py start_costmap_viewer:=true
```

This starts `pcm_costmap_viewer` as a separate test/diagnostic process. The
planner only exports a compact, subscriber-aware snapshot at most every
`gp_state_period` seconds (default 0.25 s). It reuses the inducing points and GP
coefficients from its existing evaluation, including padded object slots; it
does not evaluate a display grid or fit another GP. Publishing follows the
`start_costmap_viewer` launch argument; set it to `false` to disable both.

In the same Foxglove 3D panel as the road and trajectory, enable
`/plan/pcm_longitudinal/cost_surface`. In RViz, add a **Marker** display for
`/plan/pcm_longitudinal/cost_surface`. The marker uses `base_link`, matching the
planner, and normal RViz TF handling. A **PointCloud2** display can instead use
`/plan/pcm_longitudinal/cost_cloud`, with RGB colors or the raw `mean` field.

The surface obeys `z = road_z + surface_offset + height_scale * mean`. Negative
costs remain negative; there is no per-frame normalization or clipping. The
default 1.15 m offset keeps the usual negative costs visible above the road.
Setting `height_scale` to zero gives a road-following heatmap. Road height uses
the ego path's centerline and sampled boundary elevations, including bank where
available; outside that interpolation area it falls back to the nearest
centerline segment. This extrapolation is only a display approximation.

Viewer parameters are in the same launch file: extent (`forward`, `backward`,
`width`), `resolution`, `update_rate`, `prediction_time`, `height_scale`,
`surface_offset`, `opacity`, and `stale_timeout`. Defaults use a 65 by 20 m
extent, a 1 m grid (1,386 samples), and 4 Hz. `max_grid_points` rejects oversized
requests before allocating the grid. Grid evaluation is batched in the viewer;
its snapshot queue holds only the latest state. Each state is rendered once,
and markers expire after the stale timeout.

Snapshots and rendered markers/clouds retain the original `/scenario` timestamp,
not the planner or viewer publication time. This lets Foxglove use the ego-frame
transform corresponding to the geometry's measurement time.

The displayed `mean` is the planner's combined field: smooth maximum of the
pedestrian and vehicle means (each offset by -1), combined with the minimum
of the two lane means (zero offset), using the same smooth maximum as
the optimizer. Far from all supports, lane means approach zero; smooth-max
introduces a small positive combined-field baseline of approximately 0.0026
with the current smoothing parameter. Color uses a fixed blue/teal-to-yellow/red scale. This is not
collision probability or a normalized cost. For debugging, the point cloud
also carries `road_z` and the actual `prediction_time`.

By default, the requested time (0.5 s) snaps to the nearest planner prediction
sample; disable `snap_to_prediction_time` to inspect the exact requested time.
Time affects only dynamic-object kernels. Lane fitting and evaluation share
the same XY-only kernel in the planner and viewer, using coordinate differences
directly. The min/smooth-max combination, offsets and optimizer weights remain
unchanged. GP snapshots use version 2, with XY lane supports and XY-time dynamic
supports; restart both nodes after rebuilding when upgrading from version 1.

## Run

Use a sourced Ubuntu/WSL ROS Humble environment. Dependencies include CasADi,
OSQP, NumPy, SciPy, Shapely, and a C compiler. They are installed in this
workspace's Python environment; see `requirements.txt` for another environment.

From `~/ros2_ws`, after stopping old simulator/controller instances:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select simulator plan_predictive_cost_map
source install/setup.bash
ros2 launch plan_predictive_cost_map pcm_planner.launch.py start_simulator:=true
```

This starts all four nodes, with combined planning, external tracking steering,
the occluded-pedestrian MAT scenario, 50 km/h initial speed, and full downhill
physics. The simulator holds the initial state and actor time until the first
final command. First-time solver compilation can take around two minutes; it
does not consume the scenario while waiting. The compiled solver is cached in
`~/.cache/pcm_planner`, keyed by source, settings, CasADi version, and architecture.

Override `scenario_file:=/absolute/path/to/scenario.mat` for another scene or
`config_file:=/absolute/path/to/pcm_planner.yaml` for different parameters. With
an already-running simulator, omit `start_simulator:=true`. It must use:

```text
command_mode=acceleration
vehicle_model=dynamic
use_internal_controller=false
simulation_time_stamps=true
simulation_step=0.05
use_road_gradient=true
road_pitch=0
```

The simulator's default speed-command interface remains available for legacy
nodes. The split stack uses requested acceleration before vehicle loads and
tire steering angle. The simulator independently clamps actuator inputs and
brakes on command timeout after control has started.

## Responsibilities

| Module | Responsibility |
| --- | --- |
| `pcm_planner.py` | Scenario/ego pairing, trajectory publication, objective diagnostics, GP clouds. |
| `mpc_tracker.py` | Reference/ego feedback only; publishes candidate commands. |
| `safety_supervisor.py` | Command gate, current hazards/freshness, final-command ownership. |
| `planner/optimizer.py` | Combined polynomial MPC, constraints, validation, warm start, compiled cache. |
| `planner/tracking.py` | Short-horizon linearized tracking QP using OSQP. |
| `planner/dynamics.py` | Simulator-matching dynamic model and net-acceleration/load conversion. |
| `planner/costmap.py` | Bounded GP fitting and posterior-mean evaluation. |
| `planner/costmap_visualization.py` | Fixed-scale raw component/time-slice clouds. |
| `planner/messages.py` | Frame/time conversion, branch extraction, 3D reference conversion. |
| `planner/geometry.py` | Road union, projection and oriented footprint checks. |
| `planner/worker.py` | Isolated latest-only planning process. |

Planner configuration targets 5 Hz replanning with a 3-second horizon. Its
worker has size-one queues and never executes in the tracking process. Failed
replans do not clear the last trajectory or issue braking commands. The tracker
targets 20 Hz with a 0.75-second prediction horizon. It can retain the previous
reference while fresh and long enough, but not indefinitely. New references
are checked for position/speed handoff continuity and model-time ordering.

Simulator-step timestamps, rather than wall-clock progress under CPU load,
drive trajectory execution. A new simulator epoch clears old references. The
tracker requires fresh ego feedback, a recent reference publication, enough
future horizon, and bounded tracking error. Expiry produces a persistent,
load-compensated net-braking ramp; it does not repeatedly restart from a stale
acceleration sample.

The supervisor receives current object predictions on the simulator's
lightweight `/simulator/tracking_scenario` stream, not the full road mesh. It
checks synchronized safety data, fresh candidate commands and active trajectory.
Real hazards and virtual branches above the configured joint-probability
threshold can trigger emergency braking. Lower-probability virtual branches
remain part of the planner's GP risk model. Missing/stale supervision inputs
trigger controlled stopping once control has started. No command is sent while
the tracker is still waiting for its first usable reference.

The supervisor and tracker are separate processes; scenario processing cannot
block the tracking callback. Tests verify no scenario subscription in the
tracker and no actuator publisher in the planner. This is not a certified
safety architecture or a guarantee that every emergency stop avoids collision.

## Objective Terms

The combined planner optimizes cubic net-acceleration and steering polynomials
in normalized horizon time: eight decision variables, fixed 3-second horizon.
The simulator-matching dynamic rollout uses 50 ms controls and 5 ms substeps.
Gravity, drag, rolling and lateral coupling are included. Tests compare the
CasADi update against the real C++ model. Drive requests compensate vehicle
loads so downhill holding effort is not mistaken for unwanted acceleration.

Each planner status contains a weighted `objective` component breakdown:

| Component | Weight |
| --- | --- |
| GP risk integral | `weight_safety` |
| Saturating MATLAB-style overspeed penalty | `weight_velocity * weight_overspeed` |
| Normalized underspeed error | `weight_velocity * weight_underspeed` |
| Positive net longitudinal acceleration | `weight_acceleration * weight_positive_acceleration` |
| Net braking acceleration | `weight_acceleration * weight_braking_acceleration` |
| Lateral acceleration | `weight_lateral_acceleration` |
| Steering amplitude | `weight_steering` |
| Propulsion/braking request amplitude | `weight_input_amplitude` |
| Normalized jerk and steering rate | `weight_rate` |
| Crash-threshold violation | `weight_crash` |
| Terminal GP risk | `weight_terminal` |

The MATLAB horizon-length term is constant with a fixed horizon and is omitted.
Weights/kernel parameters are configurable defaults, not numerically matched
MATLAB parameters: its external `params` structure has not been supplied.
Other deliberate differences include dynamic-model rollouts, cubic rather
than fourth-order command polynomials, bounded GP support and explicit tracking.

Steering, speed, acceleration, command rates, lateral acceleration, road
containment, route-end stopping distance and GP risk are constrained. The target
path is a cost preference, not a hard ban on lane changes. The optimizer may use
the connected drivable-road union, including opposing lanes as approved for
these research scenarios. Full-footprint polygon checks reject off-road plans.
Missing no-passing/solid-line rules and independently elevated adjacent roads
remain limitations. Geometric permission does not guarantee a local solve finds
the globally best overtaking trajectory.

## GP Semantics and Display

The six fields are vehicles, pedestrians, cyclists, obstacles, target path and
drivable road. Each uses squared-exponential kernels and a Cholesky posterior:

```text
alpha = solve(K(Z,Z) + diag(noise) + jitter, targets)
mean(query) = K(query,Z) @ alpha + baseline
combined = max(vehicle, pedestrian, cyclist, obstacle, min(target, road))
```

Matrix solves occur outside the NLP. Branch confidence `p`, existence `e` and
severity `s` follow the MATLAB GP semantics: dynamic targets are `1+p*s` with
baseline -1. Noise is `0.01+0.05*t+(e*p)*(1-e*p)*s*s` for vehicles,
`0.02+0.25*t+(e*p)*(1-e*p)*s*s` for pedestrians, and
`0.01+0.20*t+(e*p)*(1-e*p)*s*s` for cyclists. Obstacles use fixed noise 0.001.
This is not a joint-probability expected-cost replacement. Conditional branch
confidences are normalized. The default support budget is 48 per field, with
at least two points per accepted branch. Branch overflow is reported, not
silently dropped; support thinning is still an approximation needing validation.

In Foxglove, enable one of `/plan/pcm/costmap/t0`, `/t1`, `/t2`, `/t3` as a point
cloud with fixed frame `map` and packed `rgb` colors. These are 0, 1, 2 and 3
seconds ahead, capped by the horizon. Use the exact `prediction_time` point field
when changing horizons. Road Z is retained with a small display offset. Initially
show one slice to avoid overlapping surfaces. The existing target-path and
`/trajectoryVisualization` topics provide overlays.

Raw point fields include `cost`, `vehicle`, `pedestrian`, `cyclist`, `obstacle`,
`target`, `road`, `risk_limit`, and `prediction_time`. Blue corresponds to -1,
yellow near the default 0.05 risk limit, and red near 3. Colors do not
auto-normalize between frames. These are GP costs, not collision probabilities.
Inspect component fields to identify which field dominates an unexpected stop.

Display defaults are 1 Hz and a 1 m grid; configure `visualize_costmap`,
`costmap_debug_rate`, and `costmap_resolution`. Clouds are generated outside the
tracker and remain available for fitted GP data when optimization rejects a
plan. First compilation must finish before the worker fits its first map.
Clouds retain their source snapshot stamp; a frozen debug view is not fresh data.

## Diagnostics and Validation

- `/plan/pcm_planner/status`: `trajectory_published`, `replan_failed`, fit/solve
  time, solver status, and weighted objective components.
- `/control/mpc_tracker/status`: `command_valid`, solve time, tracking errors,
  reference age and remaining horizon, or controlled-stop reason.
- `/control/safety_supervisor/status`: `pass_through`, `controlled_stop`, or
  `emergency_brake`, with `intervening` and a reason.

Use one BLAS/OpenMP thread for predictable runtime. The code supplies defaults
when environment settings are absent. This ROS/Python/WSL/IPOPT/OSQP system is
soft-real-time; measured rates are not worst-case or stability guarantees.

From the package directory in a sourced shell:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest \
  test/test_pcm_planner.py test/test_mpc_tracker.py test/test_costmap_visualization.py \
  test/test_pcm_planner_ros.py test/test_mpc_tracker_ros.py \
  test/test_safety_supervisor_ros.py -q -s
```

Tracker-only tests use generated references with no GP planner, including
50 km/h flat/uphill/steep-downhill tracking, state-error correction, reference
reuse and expiry. Integration tests use the actual four-node stack, GP clouds,
objective diagnostics, and sole final-command ownership. A no-planner simulator
test measured about 50 ms command spacing and roughly 7-10 ms tracking p95;
four-node tests measured about 50 ms spacing with tracking p95 around 9-15 ms
on this machine. These do not prove behavior in every scene.

See `ACCEPTANCE.md` and the reports under `test/` for full driving gates. The
historical seven-scenario baseline remains zero passes and is not overwritten
by component-test results. The new supervised pedestrian smoke report still
fails jerk: 12.51 m/s^3 at 2.35 seconds against the 6 m/s^3 limit. Its node split
works, but full pedestrian/overtaking acceptance has not been established.