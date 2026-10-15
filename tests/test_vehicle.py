import numpy as np
import pytest

from enginetwin.vehicle import VehicleSpec, choose_gear, demand_from_trace, engine_point, wheel_power_w

SPEC = VehicleSpec()
V100 = 100 / 3.6  # m/s


def test_the_default_vehicle_is_the_assumed_kia_soul():
    assert (SPEC.mass_kg, SPEC.final_drive, SPEC.wheel_radius_m, SPEC.crr, SPEC.cd_a) == (1300.0, 4.1, 0.31, 0.012, 0.84)
    assert SPEC.gear_ratios == (3.62, 1.96, 1.30, 0.97, 0.80)
    assert (SPEC.drivetrain_efficiency, SPEC.air_density, SPEC.idle_rpm, SPEC.redline_rpm) == (0.9, 1.2, 800.0, 6000.0)


def test_steady_100_kmh_needs_rolling_resistance_plus_air_drag_times_speed():
    # rolling 0.012 x 1300 x 9.81 = 153.04 N; air 0.5 x 1.2 x 0.84 x 27.78^2 = 388.89 N; (153.04 + 388.89) x 27.78 = 15054 W
    assert wheel_power_w(100, 0.0, SPEC) == pytest.approx(15054, rel=0.001)


def test_accelerating_adds_mass_times_acceleration_times_speed():
    # 20 km/h = 5.556 m/s at 1 m/s^2: inertia 1300 x 1 x 5.556 = 7222 W, rolling 153.04 x 5.556 = 850 W, air 0.504 x 5.556^3 = 86 W
    assert wheel_power_w(20, 1.0, SPEC) == pytest.approx(7222 + 850 + 86, rel=0.002)


def test_a_vehicle_at_rest_needs_no_power():
    assert wheel_power_w(0, 0.0, SPEC) == 0.0


def test_engine_speed_and_torque_follow_from_gear_ratio_and_wheel_size():
    # 5th gear at 100 km/h: wheel 27.78 / 0.31 = 89.61 rad/s; engine 89.61 x 0.80 x 4.1 = 293.9 rad/s = 2807 rpm.
    # Engine power = 15054 / 0.9 = 16727 W, so torque = 16727 / 293.9 = 56.9 Nm.
    rpm, torque = engine_point(100, gear=5, power_w=15054, spec=SPEC)

    assert rpm == pytest.approx(2807, rel=0.001) and torque == pytest.approx(56.9, rel=0.002)


def test_a_lower_gear_spins_the_engine_faster_for_the_same_road_speed():
    rpms = [engine_point(100, g, 15054, SPEC)[0] for g in (1, 2, 3, 4, 5)]

    assert rpms == sorted(rpms, reverse=True)
    assert rpms[0] / rpms[4] == pytest.approx(3.62 / 0.80)


def test_negative_power_on_the_overrun_gives_no_torque_and_standstill_idles_at_idle_speed():
    assert engine_point(60, 4, -5000, SPEC)[1] == 0.0
    assert engine_point(0, 1, 0, SPEC) == (800.0, 0.0)


def test_cruising_picks_the_highest_gear_that_keeps_the_engine_above_1500_rpm():
    # 100 km/h in 5th is 2807 rpm. At 20 km/h even 2nd gear (1372 rpm) is below 1500; 1st gives 2535 rpm.
    assert choose_gear(100, wheel_power_w(100, 0, SPEC), SPEC) == 5
    assert choose_gear(20, wheel_power_w(20, 0, SPEC), SPEC) == 1


def test_hard_acceleration_forces_a_downshift_until_the_torque_fits_under_the_full_load_curve():
    # 50 km/h at 2 m/s^2 needs about 39.6 kW at the wheels, 44.0 kW at the engine. 4th gear (1702 rpm) would need 247 Nm
    # and 3rd (2280 rpm) 184 Nm, both above 90% of the full-load torque; 2nd (3437 rpm) needs 122 Nm and fits.
    assert choose_gear(50, wheel_power_w(50, 2.0, SPEC), SPEC) == 2


def test_the_demand_table_turns_a_speed_trace_into_acceleration_power_gear_rpm_and_torque():
    trace = 3.6 * np.arange(10)  # 0, 3.6, 7.2 ... km/h: exactly 1 m/s^2 when sampled every second
    d = demand_from_trace(trace, dt=1.0, spec=SPEC)

    assert list(d.columns) == ["speed_kmh", "accel_ms2", "wheel_power_w", "gear", "rpm", "torque_nm", "engine_power_w"]
    assert np.allclose(d.accel_ms2, 1.0) and len(d) == 10
    assert d.rpm.min() >= 800 and d.rpm.max() <= 6000
    assert np.allclose(d.engine_power_w, d.torque_nm * d.rpm * 2 * np.pi / 60)


def test_a_steady_cruise_trace_stays_in_one_gear_at_the_hand_calculated_point():
    d = demand_from_trace(np.full(20, 100.0), dt=1.0, spec=SPEC)

    assert set(d.gear) == {5} and d.rpm.iloc[5] == pytest.approx(2807, rel=0.001) and d.torque_nm.iloc[5] == pytest.approx(56.9, rel=0.01)
