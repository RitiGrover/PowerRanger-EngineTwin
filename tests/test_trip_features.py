import numpy as np
import pandas as pd
import pytest

from enginetwin.features import trip_features


def trips(**cols):
    return pd.DataFrame(cols)


def test_high_load_share_is_the_fraction_of_time_above_80_percent_load():
    df = trips(trip_id=["A-0"] * 4, calculated_load_value=[50.0, 81.0, 95.0, 80.0])  # exactly 80 is not above

    assert trip_features(df)["high_load_share"].iloc[0] == 0.5


def test_aggressiveness_is_the_average_pedal_change_per_second():
    # Pedal 0 -> 10 -> 4 -> 4: changes of 10, 6 and 0 -> mean 16/3 percentage points per second.
    df = trips(trip_id=["A-0"] * 4, accelerator_pedal_value=[0.0, 10.0, 4.0, 4.0])

    assert trip_features(df)["aggressiveness"].iloc[0] == pytest.approx(16 / 3)


def test_warmup_time_is_seconds_until_coolant_first_reaches_85_degrees():
    df = trips(
        trip_id=["A-0"] * 6 + ["B-0"] * 2,
        time_s=[1, 2, 3, 4, 5, 6, 1, 2],
        engine_coolant_temperature=[60.0, 70.0, 84.0, 85.0, 90.0, 90.0, 40.0, 50.0],
    )

    out = trip_features(df).set_index("trip_id")

    assert out.loc["A-0", "warmup_time_s"] == 3  # reached 85 at t=4, the trip started at t=1
    assert out.loc["A-0", "warm_share"] == 0.5
    assert np.isnan(out.loc["B-0", "warmup_time_s"])  # never got warm
    assert out.loc["B-0", "warm_share"] == 0.0


def test_idle_rpm_stability_is_the_spread_of_engine_speed_while_standing_still():
    # Standing with the engine running at 900, 1000 and 800 rpm -> spread (sample std) of 100. Moving rows ignored.
    df = trips(
        trip_id=["A-0"] * 5,
        vehicle_speed=[0, 0, 0, 50, 0],
        engine_speed=[900.0, 1000.0, 800.0, 3000.0, 0.0],  # the last standing row has the engine off
    )

    assert trip_features(df)["idle_rpm_std"].iloc[0] == pytest.approx(100.0)


def test_trip_fuel_economy_is_distance_driven_over_fuel_used():
    # 4 seconds at 36 km/h = 0.04 km. Fuel: 4 x 1000 uL = 0.004 L.  0.04 / 0.004 = 10 km/L.
    df = trips(trip_id=["A-0"] * 4, vehicle_speed=[36.0] * 4, fuel_consumption=[1000.0] * 4)

    assert trip_features(df)["fuel_economy_km_per_l"].iloc[0] == pytest.approx(10.0)


def test_fuel_economy_is_unknown_when_a_trip_has_no_fuel_data():
    df = trips(trip_id=["A-0"] * 3, vehicle_speed=[30.0] * 3, fuel_consumption=[np.nan] * 3)

    assert np.isnan(trip_features(df)["fuel_economy_km_per_l"].iloc[0])


def test_fuel_economy_only_counts_distance_driven_while_fuel_was_known():
    # 4 s at 36 km/h. Fuel is only known for the first 2 s: 0.02 km over 0.002 L = 10 km/L (not 20).
    df = trips(
        trip_id=["A-0"] * 4,
        time_s=[0, 1, 2, 3],
        vehicle_speed=[36.0] * 4,
        fuel_consumption=[1000.0, 1000.0, np.nan, np.nan],
    )

    assert trip_features(df)["fuel_economy_km_per_l"].iloc[0] == pytest.approx(10.0)


def test_totals_weight_each_sample_by_the_time_it_covers():
    # Samples every 0.5 s. Standing for 3 samples at 400 uL/s: 3 * 400 * 0.5 s = 600 uL = 0.6 mL idle waste.
    # Whole trip: fuel 4 * 400 * 0.5 = 800 uL = 0.0008 L; distance 30 km/h * 0.5 s = 0.004167 km -> 5.208 km/L.
    df = trips(
        trip_id=["A-0"] * 4,
        time_s=[0.0, 0.5, 1.0, 1.5],
        vehicle_speed=[0, 0, 0, 30],
        fuel_consumption=[400.0] * 4,
    )

    out = trip_features(df).iloc[0]

    assert out["idle_fuel_ml"] == pytest.approx(0.6)
    assert out["fuel_economy_km_per_l"] == pytest.approx(5.2083, abs=1e-3)


def test_idle_fuel_waste_is_fuel_burned_while_the_car_is_standing_still():
    # Second-by-second samples. Standing: 300 + 500 uL = 0.8 mL. The moving second does not count.
    df = trips(
        trip_id=["A-0"] * 3 + ["B-0"] * 2,
        vehicle_speed=[0, 0, 40, 30, 50],
        fuel_consumption=[300.0, 500.0, 1000.0, 900.0, 800.0],
    )

    out = trip_features(df).set_index("trip_id")

    assert out.loc["A-0", "idle_fuel_ml"] == pytest.approx(0.8)
    assert out.loc["B-0", "idle_fuel_ml"] == 0.0
