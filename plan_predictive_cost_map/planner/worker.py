"""Latest-only, isolated optimization worker; no ROS APIs in this process."""

from queue import Empty, Full
from time import perf_counter
import os
import tempfile


def put_latest(queue, item):
    try:
        queue.put_nowait(item)
        return
    except Full:
        pass
    try:
        queue.get_nowait()
    except Empty:
        pass
    try:
        queue.put_nowait(item)
    except Full:
        pass


def plan_snapshot(optimizer, snapshot, config):
    from .costmap import fit_costmap
    started = perf_counter()
    costmap = fit_costmap(snapshot.branches, snapshot.road.points, snapshot.lanes,
                          snapshot.state[:2], config)
    fitted_at = perf_counter()
    model_time = snapshot.stamp_ns * 1e-9 if hasattr(snapshot, 'stamp_ns') else snapshot.received_at
    previous_time = getattr(optimizer, 'snapshot_time', model_time - config.control_step)
    try:
        plan = optimizer.solve(snapshot.state, snapshot.previous_command, snapshot.road,
                               costmap, snapshot.branches, max(0., model_time - previous_time))
    except Exception as error:
        elapsed = perf_counter() - started
        return {'error': str(error), 'version': snapshot.version, 'snapshot': snapshot, 'costmap': costmap,
                'fit_ms': (fitted_at - started) * 1000, 'total_ms': elapsed * 1000}
    optimizer.snapshot_time = model_time
    elapsed = perf_counter() - started
    timings = {'fit_ms': (fitted_at - started) * 1000,
               'solve_ms': plan.solve_seconds * 1000,
               'validation_ms': max(0., elapsed - (fitted_at - started) - plan.solve_seconds) * 1000,
               'total_ms': elapsed * 1000}
    if elapsed > config.planning_deadline:
        return dict(timings, snapshot=snapshot, costmap=costmap,
                error='Planning pipeline exceeded its %.1f ms deadline: %.1f ms' %
                    (config.planning_deadline * 1000, elapsed * 1000),
                    version=snapshot.version, deadline_miss=True)
    return dict(timings, plan=plan, snapshot=snapshot, costmap=costmap)


def optimize_worker(incoming, outgoing, stop, config, vehicle):
    os.setsid()
    from .optimizer import PolynomialMPC
    with tempfile.TemporaryDirectory(prefix='pcm_planner_') as directory:
        os.chdir(directory)
        try:
            optimizer = PolynomialMPC(config, vehicle)
        except Exception as error:
            put_latest(outgoing, {'error': 'Solver initialization: ' + str(error)})
            return
        put_latest(outgoing, {'ready': True})
        while not stop.is_set():
            try:
                snapshot = incoming.get(timeout=.1)
            except Empty:
                continue
            started = perf_counter()
            try:
                put_latest(outgoing, plan_snapshot(optimizer, snapshot, config))
            except Exception as error:
                put_latest(outgoing, {'error': str(error), 'version': snapshot.version,
                                      'total_ms': (perf_counter() - started) * 1000})