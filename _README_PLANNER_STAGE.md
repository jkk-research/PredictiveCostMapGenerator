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