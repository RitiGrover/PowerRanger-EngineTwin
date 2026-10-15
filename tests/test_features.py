import numpy as np
import pandas as pd
import pytest

from enginetwin.features import add_instant_features


def frame(**cols):
    return pd.DataFrame(cols)


def test_torque_utilisation_is_torque_over_maximum_indicated_torque():
    df = frame(engine_torque=[30.0, 60.0], maximum_indicated_engine_torque=[60.0, 60.0])

    out = add_instant_features(df)

    assert list(out["torque_utilisation"]) == [0.5, 1.0]


def test_mechanical_efficiency_is_one_minus_friction_over_indicated_torque():
    # Indicated torque = brake torque + friction torque = 30 + 10 = 40  ->  1 - 10/40 = 0.75
    df = frame(engine_torque=[30.0, 0.0], torque_of_friction=[10.0, 0.0])

    out = add_instant_features(df)

    assert out["mechanical_efficiency"].iloc[0] == 0.75
    assert np.isnan(out["mechanical_efficiency"].iloc[1])


def test_total_fuel_trim_adds_short_and_long_term_trims():
    df = frame(short_term_fuel_trim_bank1=[3.0, -4.0], long_term_fuel_trim_bank1=[2.5, 1.0])

    assert list(add_instant_features(df)["total_fuel_trim"]) == [5.5, -3.0]


def test_converter_slip_is_the_fraction_of_engine_speed_not_reaching_the_gearbox():
    # Engine at 2000 rpm, converter output at 1800 rpm -> 10% slip. A stopped engine has no defined slip.
    df = frame(engine_speed=[2000.0, 0.0], torque_converter_speed=[1800.0, 0.0])

    out = add_instant_features(df)

    assert out["converter_slip"].iloc[0] == pytest.approx(0.1)
    assert np.isnan(out["converter_slip"].iloc[1])


def test_gear_match_is_engine_revs_per_km_per_hour():
    # 3000 rpm at 60 km/h = 50 rpm per km/h. At a standstill the ratio is meaningless.
    df = frame(engine_speed=[3000.0, 900.0], vehicle_speed=[60.0, 0.0])

    out = add_instant_features(df)

    assert out["gear_match"].iloc[0] == 50.0
    assert np.isnan(out["gear_match"].iloc[1])


def test_instant_features_include_power_efficiency_and_fuel_per_work():
    # Same worked example as the efficiency tests: 30 Nm, 3000 rpm, 1000 uL/s.
    df = frame(engine_torque=[30.0], engine_speed=[3000.0], fuel_consumption=[1000.0])

    out = add_instant_features(df)

    assert out["brake_power_w"].iloc[0] == pytest.approx(9424.8, abs=0.1)
    assert out["thermal_efficiency"].iloc[0] == pytest.approx(0.2915, abs=2e-4)
    assert out["bsfc_g_per_kwh"].iloc[0] == pytest.approx(284.6, abs=0.2)  # 2682 g/h / 9.4248 kW


def test_efficiency_above_the_physical_limit_is_masked_as_a_measurement_artefact():
    # 30 Nm at 3000 rpm with only 200 uL/s of fuel would be 146% efficient: torque and fuel are out of step.
    df = frame(engine_torque=[30.0], engine_speed=[3000.0], fuel_consumption=[200.0])

    out = add_instant_features(df)

    assert np.isnan(out["thermal_efficiency"].iloc[0])
    assert np.isnan(out["bsfc_g_per_kwh"].iloc[0])  # masked together: the same row is untrustworthy


def test_features_whose_inputs_are_missing_are_left_out_instead_of_failing():
    out = add_instant_features(frame(engine_torque=[30.0], torque_of_friction=[10.0]))

    assert "mechanical_efficiency" in out.columns
    assert "torque_utilisation" not in out.columns  # no maximum_indicated_engine_torque column


def test_torque_utilisation_is_undefined_when_no_torque_is_available():
    df = frame(engine_torque=[0.0], maximum_indicated_engine_torque=[0.0])

    assert np.isnan(add_instant_features(df)["torque_utilisation"].iloc[0])
