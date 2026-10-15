import numpy as np
import pandas as pd
import pytest

from enginetwin.ved import load_ved

RAW = [
    "DayNum", "VehId", "Trip", "Timestamp(ms)", "Latitude[deg]", "Longitude[deg]", "Vehicle Speed[km/h]",
    "MAF[g/sec]", "Engine RPM[RPM]", "Absolute Load[%]", "Outside Air Temperature[DegC]", "Fuel Rate[L/hr]",
    "Short Term Fuel Trim Bank 1[%]", "Short Term Fuel Trim Bank 2[%]",
    "Long Term Fuel Trim Bank 1[%]", "Long Term Fuel Trim Bank 2[%]",
]


def write_week(tmp_path, rows, name="VED_171101_week.csv"):
    path = tmp_path / name
    pd.DataFrame(rows, columns=RAW).to_csv(path, index=False)
    return path


def row(veh, trip, ts_ms, speed=50.0, maf=14.7, rpm=2000.0, stft=0.0, ltft=0.0, fuel_rate=np.nan):
    return [1, veh, trip, ts_ms, 42.0, -83.0, speed, maf, rpm, 40.0, 10.0, fuel_rate, stft, np.nan, ltft, np.nan]


def test_fuel_is_estimated_from_air_flow_when_the_car_reports_no_fuel_rate(tmp_path):
    # Stoichiometric petrol: 14.7 g/s of air burns 1 g/s of fuel = 1 / 0.745 g/mL = 1342.3 uL/s.
    # With +6% short-term and +4% long-term trim the ECU is adding 10% more fuel: 1.1 g/s = 1476.5 uL/s.
    rows = [row(10, 1, 1000, maf=14.7), row(10, 1, 2000, maf=14.7, stft=6.0, ltft=4.0)]

    df = load_ved(write_week(tmp_path, rows))

    assert df["fuel_consumption"].iloc[0] == pytest.approx(1342.3, abs=0.1)
    assert df["fuel_consumption"].iloc[1] == pytest.approx(1476.5, abs=0.1)


def test_loader_keeps_only_combustion_cars_when_given_the_static_vehicle_table(tmp_path):
    static = tmp_path / "static.xlsx"
    pd.DataFrame({"VehId": [10, 11, 12], "Vehicle Type": ["ICE", "HEV", "ICE"]}).to_excel(static, index=False)
    rows = [row(10, 1, 1000), row(11, 1, 1000), row(12, 1, 1000), row(10, 2, 5000)]

    df = load_ved(write_week(tmp_path, rows), static_path=static)

    assert sorted(df["vehid"].unique()) == [10, 12]  # vehicle 11 is a hybrid
    assert len(df) == 3


def test_columns_that_differ_only_by_unit_stay_distinct(tmp_path):
    path = write_week(tmp_path, [row(10, 1, 1000)])
    df = pd.read_csv(path)
    df["Air Conditioning Power[kW]"] = 1.5
    df["Air Conditioning Power[Watts]"] = 1500.0
    df.to_csv(path, index=False)

    loaded = load_ved(path)

    assert loaded["air_conditioning_power_kw"].iloc[0] == 1.5
    assert loaded["air_conditioning_power_watts"].iloc[0] == 1500.0


def test_a_measured_fuel_rate_wins_over_the_estimate(tmp_path):
    # 3.6 L/h = 1 mL/s = 1000 uL/s, even though the air flow alone would suggest 1342 uL/s.
    rows = [row(10, 1, 1000, maf=14.7, fuel_rate=3.6), row(10, 1, 2000, maf=14.7)]

    df = load_ved(write_week(tmp_path, rows))

    assert df["fuel_consumption"].iloc[0] == pytest.approx(1000.0)
    assert df["fuel_consumption"].iloc[1] == pytest.approx(1342.3, abs=0.1)  # no measurement: falls back


def test_loader_names_columns_like_the_ocslab_loader_and_builds_trip_ids(tmp_path):
    rows = [row(10, 1, 5000), row(10, 1, 6000), row(10, 2, 9000), row(11, 1, 2000)]

    df = load_ved(write_week(tmp_path, rows))

    assert {"engine_speed", "vehicle_speed", "maf", "trip_id", "time_s"} <= set(df.columns)
    assert list(df["trip_id"]) == ["10-1", "10-1", "10-2", "11-1"]
    assert list(df["time_s"]) == [0.0, 1.0, 0.0, 0.0]  # seconds since each trip started
