import pandas as pd
import pytest

from enginetwin.efficiency_map import efficiency_map, relative_sweet_spot, sweet_spot


def test_relative_sweet_spot_places_the_best_cell_within_the_engines_own_range():
    # rpm spans 1000-3000 and torque 10-50. Best cell is rpm 2000 (halfway) and torque 40 (3/4 of the way up).
    cells = pd.DataFrame(
        {
            "rpm": [1000.0, 2000.0, 3000.0, 2000.0],
            "torque": [10.0, 40.0, 50.0, 10.0],
            "efficiency": [0.10, 0.30, 0.20, 0.12],
        }
    )

    assert relative_sweet_spot(cells) == (0.5, 0.75)


def test_map_leaves_out_cells_with_too_few_samples_to_trust():
    df = trace([(3000, 30.0, 1000.0)] * 3 + [(1000, 10.0, 500.0)])

    cells = efficiency_map(df, rpm_step=500, torque_step=10, min_samples=2)

    assert list(cells["rpm"]) == [3250]


def test_sweet_spot_is_the_most_efficient_cell():
    # 30 Nm @ 3000 rpm burning 1000 uL/s -> 0.2915; same point burning 2000 uL/s -> 0.1457
    df = trace([(3000, 30.0, 1000.0)] * 3 + [(1000, 10.0, 2000.0)] * 3)

    best = sweet_spot(efficiency_map(df, rpm_step=500, torque_step=10, min_samples=2))

    assert (best["rpm"], best["torque"]) == (3250, 35)
    assert best["efficiency"] == pytest.approx(0.2915, abs=2e-4)


def trace(rows):
    return pd.DataFrame(rows, columns=["engine_speed", "engine_torque", "fuel_consumption"])


def test_map_reports_the_median_efficiency_of_each_cell():
    # All rows sit in the rpm 3000-3500 / torque 30-40 cell. Two rows give 0.2915 efficiency
    # (30 Nm, 3000 rpm, 1000 uL/s, worked out by hand elsewhere); one outlier burns 4x the fuel.
    df = trace([(3000, 30.0, 1000.0), (3000, 30.0, 1000.0), (3000, 30.0, 4000.0)])

    cells = efficiency_map(df, rpm_step=500, torque_step=10, min_samples=1)

    assert len(cells) == 1
    cell = cells.iloc[0]
    assert (cell["rpm"], cell["torque"], cell["samples"]) == (3250, 35, 3)
    assert cell["efficiency"] == pytest.approx(0.2915, abs=2e-4)  # a mean would give 0.2186
