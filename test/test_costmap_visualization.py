"""Debug displays preserve exact GP values and do not depend on successful plans."""

from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip('sensor_msgs.msg')
from plan_predictive_cost_map.planner.config import PlannerConfig
from plan_predictive_cost_map.planner.costmap import fit_costmap
from plan_predictive_cost_map.planner.costmap_visualization import POINT_DTYPE, cost_colors, costmap_clouds
from plan_predictive_cost_map.planner.geometry import Road


def test_costmap_cloud_preserves_raw_components_and_time_slices():
    config = PlannerConfig(gp_support_points=8)
    route = np.array([[0., 0., 10.], [30., 0., 13.]])
    snapshot = SimpleNamespace(state=np.array([0., 0., 10., 0., 5., 0., 0.]),
                               stamp_ns=12_000_000_000, road=Road(route, 3.85, 10.))
    fitted = fit_costmap([], route, [route], [0, 0], config)
    clouds = costmap_clouds(fitted, snapshot, config, times=(0., 2.), forward=10., backward=1., width=4.)
    assert len(clouds) == 2
    data = np.frombuffer(clouds[0].data, dtype=POINT_DTYPE)
    query = np.column_stack([data['x'], data['y'], np.zeros(len(data))])
    np.testing.assert_allclose(data['cost'], fitted.evaluate(query, config), atol=1e-6)
    assert clouds[0].header.frame_id == 'map' and clouds[0].header.stamp.sec == 12
    assert np.min(data['z']) >= 10.
    np.testing.assert_allclose(data['risk_limit'], config.risk_limit)
    assert cost_colors(np.array([-1.]))[0] == (25 << 16 | 90 << 8 | 190)
    assert cost_colors(np.array([3.]))[0] == (180 << 16 | 30 << 8 | 50)