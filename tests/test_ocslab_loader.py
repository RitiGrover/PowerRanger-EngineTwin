import pandas as pd
import pytest

from enginetwin.ocslab import load_ocslab

RAW_COLUMNS = [
    "Time(s)",
    "Engine_speed",
    "Engine_torque",
    "Fuel_consumption",
    "Vehicle_speed",
    "Fuel_Pressure",
    "Engine_coolant_temperature",
    "Engine_coolant_temperature.1",
    "PathOrder",
    "Class",
]


def write_csv(tmp_path, rows):
    path = tmp_path / "ocslab.csv"
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(path, index=False)
    return path


def row(t, rpm=1000, torque=20.0, fuel=300.0, speed=0, cls="A", path_order=1):
    return [t, rpm, torque, fuel, speed, 0.0, 80.0, 80.0, path_order, cls]


def test_loader_drops_constant_and_duplicate_columns_but_keeps_signals(tmp_path):
    path = write_csv(tmp_path, [row(1, rpm=900), row(2, rpm=1500, torque=35.0), row(3, rpm=2100, torque=50.0)])

    df = load_ocslab(path)

    assert "fuel_pressure" not in df.columns  # constant: carries no information
    assert "engine_coolant_temperature_1" not in df.columns  # duplicate of engine_coolant_temperature
    assert {"engine_speed", "engine_torque", "driver", "time_s"} <= set(df.columns)


def test_loader_splits_trips_where_the_clock_resets_within_each_driver(tmp_path):
    rows = [
        row(1, 900, cls="A"), row(2, 1200, cls="A"), row(3, 1500, cls="A"),
        row(1, 1000, cls="A"), row(2, 1400, cls="A"),  # clock reset: a second trip for A
        row(1, 800, cls="B"), row(2, 1700, cls="B"),   # new driver: its own trip
    ]
    df = load_ocslab(write_csv(tmp_path, rows))

    assert list(df["trip_id"]) == ["A-0", "A-0", "A-0", "A-1", "A-1", "B-0", "B-0"]
    for _, trip in df.groupby("trip_id"):
        assert trip["time_s"].is_monotonic_increasing


def test_loader_gives_clean_snake_case_column_names(tmp_path):
    path = write_csv(
        tmp_path,
        [row(1, 900, 10.0, 250.0), row(2, 1500, 35.0, 600.0), row(3, 2100, 50.0, 900.0)],
    )

    df = load_ocslab(path)

    assert {"time_s", "engine_speed", "engine_torque", "fuel_consumption", "driver"} <= set(df.columns)
