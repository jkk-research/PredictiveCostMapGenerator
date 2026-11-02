# Driving acceptance, including cost-driven overtaking

Overtaking is a possible trajectory of the same GP-cost MPC, not a separate
behavior that overrides the optimizer. The target path is a preference; the
permitted drivable surface and collision checks are constraints. An excursion
into an adjacent/opposing lane is allowed in these research scenarios when the
costs and predictions support it. Complete solid-line/no-passing rules are not
provided in the MAT files, so this is an explicit scenario assumption, not a
claim of road-traffic-law compliance.

The former target-only hard corridor made overtaking impossible. The planner
now computes connected lateral road envelopes from the union of the published
road polygons. The optimizer can cross internal lane boundaries, and a separate
whole-footprint polygon check rejects paths outside the road union. Rear caps
created by local corridor clipping are extended for the initial ego footprint.
Road elevations still follow the target route; independently elevated adjacent
or stacked roads are not interchangeable passing surfaces.

This makes overtaking geometrically available; it does not guarantee that a
local nonlinear solve finds the globally best passing trajectory. A stationary
vehicle is not automatically a successful outcome. Progress and return toward
the target route are acceptance requirements too.

## Gates

`test/scenario_acceptance.py` discovers every MAT file in `scenario_files`.
It starts the real simulator and planner on an isolated ROS domain, with virtual
objects enabled and external direct-acceleration control. The compiled solver
is prepared before scenario time starts. Actor positions, dimensions, existence
and road geometry for collision checking come independently from the MAT file,
not from the occlusion-filtered `/scenario` object list.

| Criterion | Default threshold |
| --- | --- |
| Finite state | No NaN/infinite ego states. |
| Collision | No overlapping physical actor footprints at matching simulated samples; elevation ranges must overlap. |
| Road containment | Whole ego footprint remains in the drivable union, with 0.15 m geometry tolerance. |
| Speed | No more than 0.5 m/s above the local scenario limit. |
| Curve handling | Lateral/centripetal acceleration at most 3.5 m/s^2; target curvature also checked near the target centerline. |
| Longitudinal comfort | Acceleration between -5.5 and +3.5 m/s^2. |
| Smoothness | Measured longitudinal jerk at most 6 m/s^3. |
| Lane discipline | No unjustified target-lane excursion sustained longer than 3 seconds. |
| Stop/restart | No stop followed by restart without a nearby real or virtual hazard. |
| Clear-road progress | No 15-second period with less than 1 m progress on a clear road. |
| Pedestrian yielding | Stop before an occupied target-path pedestrian conflict, not merely avoid overlap. |
| Completion | Reach at least 90% of target-route arc length. |
| Passing recovery | Be within 0.75 m of the target route when reaching the completion region. |

These thresholds are explicit test defaults in `Limits`, not inferred after a
run to make it pass. Current cases request initial speed 5 m/s and check that
the simulator's first observation matches it. `--initial-speed` overrides this;
the simulator launch default itself is unchanged. Comfort checks start 0.5
seconds after the first valid plan; safety and road checks apply from the first
sample. A failure during preparation is not silently counted as a driving pass.

The checks use 50 ms simulator sample ordering to select the corresponding MAT
truth sample. This requires the suite's fixed simulator time step, one-times
real-time setting, and a complete ego stream. Swept-volume collision checking,
traffic-law validation and probabilistic guarantees are not supplied by this
initial suite. Real/virtual hazard proximity is a conservative explanation for
waiting or an excursion, not proof that every chosen maneuver is optimal.

## Recorded Baseline

The saved run in `test/scenario_acceptance_report.json` has **zero passing
driving scenarios**. This must not be confused with passing mathematical,
message-conversion, or acceptance-oracle unit tests.

| Scenario | First failure | Simulated time |
| --- | --- | --- |
| Highway entry | Clear-road stagnation | 33.15 s |
| Highway other vehicle entering | Smoothness, 92.39 m/s^3 jerk | 2.05 s |
| Occluded pedestrian | Smoothness, 50.16 m/s^3 jerk | 3.50 s |
| Occluded pedestrian crossing | Smoothness, 118.41 m/s^3 jerk | 2.90 s |
| Protected left with oncoming traffic | Smoothness, 29.86 m/s^3 jerk | 2.50 s |
| Roundabout with conflicting traffic | Unjustified stop/restart | 2.80 s |
| Roundabout with traffic and pedestrians | Unjustified stop/restart | 2.90 s |

Each case stops at its first definite failure by default. Requirements not
reached after an early failure are reported as not evaluated, not as passes.
In particular, this run does not validate complete overtaking, pedestrian
yielding, or crash-free completion. The initial pre-optimization baseline also
observed a highway-entry collision while the planner remained stalled; an
earlier test stop in the recorded baseline does not establish crash avoidance.

## Run

In a sourced WSL shell, build the simulator and `plan_predictive_cost_map`, then
from `src/PredictiveCostMapGenerator`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 test/scenario_acceptance.py --report test/scenario_acceptance_report.json
```

The command returns nonzero if any scenario fails. For one scenario:

```bash
python3 test/scenario_acceptance.py --scenario protected_left --report /tmp/protected_left.json
```

`--continue-after-failure` observes later hazards despite prior failures.
`--maximum-seconds` is useful for diagnosis, but a truncated run without the
required progress cannot pass. `--skip-warmup` is only for an already prepared
solver cache. Full runs can last several minutes per scene.

For pytest gates:

```bash
python3 -m pytest test/test_scenario_acceptance.py -q
PCM_SCENARIO_ACCEPTANCE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m pytest test/test_scenario_acceptance.py -q
```

The first command checks the oracle and fixture parsing, skipping the seven
long driving cases. The second runs them as real failing/passing tests, with
shared solver warmup. They are not marked expected failures or weakened to pass
the current planner. The JSON report includes limits, initial conditions,
failure time/detail, coverage, metrics and fallback categories for each case.