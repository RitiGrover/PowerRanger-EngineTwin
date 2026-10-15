import numpy as np
import pandas as pd
import pytest

from enginetwin.replay import gear_from_rpm, replay_trip
from enginetwin.vehicle import VehicleSpec

SPEC = VehicleSpec()
COLUMNS = [
    "time_s", "speed_kmh", "actual_gear", "opt_gear", "actual_rpm", "opt_rpm", "actual_eff", "opt_eff",
    "actual_fuel_l_h", "opt_fuel_l_h",
]


class PeakAt2500:
    """Stand-in for the learned model: efficiency 0.35 at 2500 rpm, falling 0.00005 per rpm either side."""

    def predict(self, X):
        return 0.35 - 0.00005 * np.abs(np.asarray(X)[:, 0] - 2500.0)


def test_the_actual_gear_is_the_one_whose_engine_speed_at_that_road_speed_is_closest_to_the_measured_rpm():
    # 100 km/h gives 2807 rpm in 5th, 3403 in 4th, 4561 in 3rd (hand calculation in the vehicle tests)
    assert [gear_from_rpm(r, 100, SPEC) for r in (2800, 3400, 4600, 6500)] == [5, 4, 3, 2]
    assert gear_from_rpm(1000, 0, SPEC) == 1  # standing still: first gear


def steady_100(n=5, source="ocslab"):
    """Cruising at 100 km/h in 3rd gear (4561 rpm). Engine power 15054 / 0.9 = 16727 W, so torque = 35.0 Nm; the measured
    fuel is 2000 uL/s = 7.2 L/h."""
    rpm = 2806.7 * 1.3 / 0.8
    trip = pd.DataFrame(
        {"time_s": np.arange(n, dtype=float), "vehicle_speed": 100.0, "engine_speed": rpm, "fuel_consumption": 2000.0}
    )
    if source == "ocslab":
        trip["engine_torque"] = 16727.0 / (rpm * 2 * np.pi / 60)
    return trip


@pytest.mark.parametrize("source", ["ocslab", "ved"])
def test_a_steady_cruise_in_the_wrong_gear_is_replayed_with_the_hand_calculated_saving(source):
    # Model efficiency: 3rd, 4561 rpm -> 0.35 - 0.00005 x 2061 = 0.24695. Best is 5th, 2807 rpm -> 0.33466.
    # Optimised fuel = 7.2 x 0.24695 / 0.33466 = 5.313 L/h. Over 5 s the saving is (7.2 - 5.313) / 3600 x 5 = 0.002621 L.
    table, s = replay_trip(steady_100(source=source), PeakAt2500(), SPEC, source)

    assert list(table.columns) == COLUMNS and len(table) == 5
    assert table.actual_gear.iloc[0] == 3 and table.opt_gear.iloc[0] == 5
    assert table.actual_eff.iloc[0] == pytest.approx(0.24695, abs=0.001) and table.opt_eff.iloc[0] == pytest.approx(0.33466, abs=0.001)
    assert table.opt_fuel_l_h.iloc[0] == pytest.approx(5.313, rel=0.01) and table.actual_fuel_l_h.iloc[0] == pytest.approx(7.2)
    assert s["total_actual_l"] == pytest.approx(7.2 / 3600 * 5, rel=0.001)
    assert s["saved_l"] == pytest.approx(0.002621, rel=0.02) and s["saved_pct"] == pytest.approx(26.2, abs=0.5)
    assert s["rupees_saved"] == pytest.approx(s["saved_l"] * 105)
    assert s["mean_actual_eff"] == pytest.approx(0.24695, abs=0.001) and s["mean_opt_eff"] == pytest.approx(0.33466, abs=0.001)


def test_the_price_per_litre_is_a_parameter():
    _, s = replay_trip(steady_100(), PeakAt2500(), SPEC, "ocslab", price=200)

    assert s["rupees_saved"] == pytest.approx(s["saved_l"] * 200)


def test_seconds_with_no_brake_power_keep_their_measured_fuel_and_cannot_be_improved():
    trip = pd.concat([steady_100(3), pd.DataFrame({"time_s": [3.0, 4.0], "vehicle_speed": 0.0, "engine_speed": 800.0, "engine_torque": 0.0, "fuel_consumption": 300.0})], ignore_index=True)
    table, s = replay_trip(trip, PeakAt2500(), SPEC, "ocslab")

    idle = table.iloc[3:]
    assert idle.actual_eff.isna().all() and idle.opt_eff.isna().all()
    assert np.allclose(idle.opt_fuel_l_h, idle.actual_fuel_l_h) and np.allclose(idle.actual_fuel_l_h, 300 * 3.6e-3)
    assert s["total_actual_l"] == pytest.approx((7.2 * 3 + 1.08 * 2) / 3600, rel=0.001)  # 300 uL/s = 1.08 L/h


def test_a_fault_raises_the_fuel_burnt_by_its_assumed_loss_and_reports_the_extra():
    # misfire 0.8: loss 0.25 x 0.8 = 20%, so the same work takes 1 / 0.8 = 1.25 times the fuel
    _, healthy = replay_trip(steady_100(), PeakAt2500(), SPEC, "ocslab")
    table, faulty = replay_trip(steady_100(), PeakAt2500(), SPEC, "ocslab", fault="misfire", severity=0.8)

    assert faulty["total_actual_l"] == pytest.approx(1.25 * healthy["total_actual_l"], rel=0.001)
    assert faulty["saved_l"] == pytest.approx(1.25 * healthy["saved_l"], rel=0.001)
    assert faulty["fault_extra_l"] == pytest.approx(0.25 * healthy["total_actual_l"], rel=0.001)
    assert table.actual_eff.iloc[0] == pytest.approx(0.8 * 0.24695, abs=0.001)
    assert healthy["fault_extra_l"] == 0.0


def test_rows_with_missing_readings_are_dropped_and_their_seconds_are_not_counted():
    trip = steady_100(4)
    trip.loc[1, "fuel_consumption"] = np.nan
    table, s = replay_trip(trip, PeakAt2500(), SPEC, "ocslab")

    assert len(table) == 3 and np.isfinite(table.opt_fuel_l_h).all()
    assert s["total_actual_l"] == pytest.approx(7.2 / 3600 * 3, rel=0.001)  # three measured seconds


def test_an_unknown_source_is_refused():
    with pytest.raises(ValueError):
        replay_trip(steady_100(), PeakAt2500(), SPEC, "simulated")
