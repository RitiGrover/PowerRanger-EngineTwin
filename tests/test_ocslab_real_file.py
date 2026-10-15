from pathlib import Path

import pytest

from enginetwin.ocslab import load_ocslab

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "osclab"
CSVS = sorted(DATA_DIR.glob("*.csv"))

pytestmark = pytest.mark.skipif(not CSVS, reason="Ocslab CSV not downloaded")


@pytest.fixture(scope="module")
def df():
    return load_ocslab(CSVS[0])


def test_real_file_has_ten_drivers_and_every_row(df):
    assert len(df) == 94380  # row count verified by hand on the downloaded file
    assert df["driver"].nunique() == 10


def test_real_file_trips_never_run_backwards_in_time(df):
    for _, trip in df.groupby("trip_id"):
        assert trip["time_s"].is_monotonic_increasing


def test_real_file_drops_dead_and_redundant_columns(df):
    for gone in ("fuel_pressure", "glow_plug_control_request", "engine_torque_after_correction", "engine_coolant_temperature_1"):
        assert gone not in df.columns


def test_real_file_sweet_spot_is_where_a_small_petrol_engine_is_efficient(df):
    from enginetwin.efficiency_map import efficiency_map, sweet_spot

    best = sweet_spot(efficiency_map(df))

    assert 1500 <= best["rpm"] <= 4000  # mid revs, not idle and not near the limiter
    assert best["torque"] >= 20  # well above idle load
    assert 0.10 <= best["efficiency"] <= 0.40  # believable for a small petrol engine


def test_real_file_trip_features_are_physically_plausible(df):
    from enginetwin.features import add_instant_features, trip_features

    feats = trip_features(add_instant_features(df))

    assert len(feats) == df["trip_id"].nunique()
    assert feats["fuel_economy_km_per_l"].between(3, 30).all()  # no car does better than 30 km/L or worse than 3
    assert (feats["idle_fuel_ml"] >= 0).all()
    assert feats["high_load_share"].between(0, 1).all()
    assert (feats["aggressiveness"] >= 0).all()


def test_real_file_instant_features_stay_in_physical_range(df):
    from enginetwin.features import add_instant_features

    out = add_instant_features(df)

    assert out["thermal_efficiency"].dropna().between(0, 0.45).all()  # petrol engines never beat ~40% brake efficiency
    assert out["torque_utilisation"].dropna().between(0, 1.5).all()
    assert out["total_fuel_trim"].abs().max() < 40  # beyond that the ECU would have thrown a fault code


def test_real_file_keeps_the_signals_the_features_need(df):
    needed = {
        "engine_speed", "engine_torque", "torque_of_friction", "maximum_indicated_engine_torque",
        "calculated_load_value", "fuel_consumption", "vehicle_speed", "current_spark_timing",
        "current_gear", "engine_coolant_temperature",
    }
    assert needed <= set(df.columns)
