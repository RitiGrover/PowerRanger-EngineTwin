import numpy as np
import pytest

from enginetwin.efficiency import brake_power_w, brake_thermal_efficiency, bsfc_g_per_kwh, fuel_economy_km_per_l


def test_fuel_economy_is_distance_per_litre():
    # 60 km/h while burning 1000 uL/s = 3.6 L/h  ->  60 / 3.6 = 16.67 km/L
    assert fuel_economy_km_per_l(speed_kmh=60.0, fuel_ul_per_s=1000.0) == pytest.approx(16.667, abs=0.01)


def test_fuel_economy_is_undefined_when_no_fuel_is_burned():
    assert np.isnan(fuel_economy_km_per_l(speed_kmh=60.0, fuel_ul_per_s=0.0))


def test_bsfc_is_fuel_grams_per_hour_over_power_in_kilowatts():
    # Power: 31.831 Nm at 3000 rpm = 10 kW. Fuel: 1000 uL/s = 0.745 g/s = 2682 g/h. 2682 / 10 = 268.2 g/kWh.
    assert bsfc_g_per_kwh(torque_nm=31.831, rpm=3000.0, fuel_ul_per_s=1000.0) == pytest.approx(268.2, abs=0.2)


def test_bsfc_is_undefined_when_the_engine_delivers_no_power():
    assert np.isnan(bsfc_g_per_kwh(torque_nm=0.0, rpm=900.0, fuel_ul_per_s=300.0))


def test_thermal_efficiency_is_power_out_over_fuel_energy_in():
    # Power: 30 Nm * 314.159 rad/s = 9,424.8 W.
    # Fuel: 1000 uL/s = 1 mL/s * 0.745 g/mL = 0.745 g/s = 0.000745 kg/s * 43.4 MJ/kg = 32,333 W.
    eff = brake_thermal_efficiency(torque_nm=30.0, rpm=3000.0, fuel_ul_per_s=1000.0)
    assert eff == pytest.approx(0.2915, abs=2e-4)


def test_thermal_efficiency_is_undefined_when_no_fuel_is_burned():
    assert np.isnan(brake_thermal_efficiency(torque_nm=30.0, rpm=3000.0, fuel_ul_per_s=0.0))


def test_brake_power_is_torque_times_angular_speed():
    # 100 Nm at 3000 rpm: 100 * (3000 * 2*pi / 60) = 100 * 314.159 rad/s ~ 31,416 W
    assert brake_power_w(torque_nm=100.0, rpm=3000.0) == pytest.approx(31415.9, abs=0.1)


def test_brake_power_is_zero_when_the_engine_is_stopped():
    assert brake_power_w(torque_nm=50.0, rpm=0.0) == 0.0
