import numpy as np
import pytest

from enginetwin.optimiser import candidates, knock_risk, learned_efficiency, optimise
from enginetwin.vehicle import VehicleSpec, wheel_power_w

SPEC = VehicleSpec()
P100 = wheel_power_w(100, 0.0, SPEC)  # 15054 W at the wheels, steady 100 km/h


class PeakAt2500:
    """Stand-in for the learned model: efficiency 0.35 at 2500 rpm, falling 0.00005 per rpm either side."""

    def __init__(self, peak=2500.0):
        self.peak, self.seen = peak, None

    def predict(self, X):
        self.seen = np.asarray(X)
        return 0.35 - 0.00005 * np.abs(self.seen[:, 0] - self.peak)


def test_the_model_is_asked_in_the_feature_order_it_was_trained_on_with_load_as_a_share_of_157_nm():
    model = PeakAt2500()
    eff = learned_efficiency(model, rpm=2000.0, torque_nm=78.5, speed_kmh=50.0, spec=SPEC)

    # engine_speed, load %, vehicle_speed, rpm per km/h (2000 / 50), total fuel trim
    assert model.seen.tolist() == [[2000.0, 50.0, 50.0, 40.0, 0.0]]  # 78.5 / 157 = 50%
    assert eff == pytest.approx(0.35 - 0.00005 * 500)


def test_knock_risk_is_high_only_at_heavy_load_below_2500_rpm():
    # Full-load torque at 2000 rpm is 135 Nm (halfway between 125 at 1500 and 145 at 2500); 85% of it is 114.75 Nm.
    assert knock_risk(2000, 120.0, SPEC) and not knock_risk(2000, 100.0, SPEC)
    assert not knock_risk(3000, 150.0, SPEC)  # the same heavy load is fine above 2500 rpm


def test_candidates_at_100_kmh_are_the_three_gears_that_stay_under_the_redline():
    # Engine speeds 2807 (5th), 3403 (4th), 4561 (3rd) rpm. 2nd would need 6877 and 1st 10168: over 6000.
    found = candidates(100, P100, SPEC)

    assert [c["gear"] for c in found] == [5, 4, 3]
    assert [round(c["rpm"]) for c in found] == [2807, 3403, 4561]
    assert [round(c["torque_nm"], 1) for c in found] == [56.9, 46.9, 35.0]


def test_a_gear_that_needs_more_than_95_percent_of_full_load_torque_is_not_a_candidate():
    # 50 km/h at 2 m/s^2 needs 44 kW at the engine. 4th gear (1702 rpm) would take 247 Nm, 3rd 184 Nm; 2nd takes 122 Nm.
    found = candidates(50, wheel_power_w(50, 2.0, SPEC), SPEC)

    assert [c["gear"] for c in found] == [2]


def test_a_gear_sitting_at_97_percent_of_full_load_is_over_the_95_percent_limit():
    # 50 km/h, 47.07 kW at the wheels = 52.3 kW at the engine. 2nd gear turns 3437 rpm (359.9 rad/s): 145.3 Nm against a
    # full-load 149.8 Nm there (97%). Above 2500 rpm knock is not a concern, so only the torque limit rules it out.
    assert candidates(50, 47070.0, SPEC) == []


def test_a_tighter_knock_limit_removes_the_heavily_loaded_low_speed_gears():
    # 50 km/h with 18.4 kW at the engine: 4th gear runs 1702 rpm at 103 Nm, 80% of the 129 Nm full load there.
    power = 16555.0
    assert 4 in [c["gear"] for c in candidates(50, power, SPEC)]  # 80% is under the 85% limit
    assert 4 not in [c["gear"] for c in candidates(50, power, SPEC, knock_limit=0.6)]


def test_the_optimiser_picks_the_gear_with_the_best_learned_efficiency_and_reports_the_gain_over_the_actual_gear():
    r = optimise(100, P100, PeakAt2500(), SPEC, actual_gear=3)

    # 5th: 2807 rpm -> 0.35 - 0.00005 x 307 = 0.3347. 3rd: 4561 rpm -> 0.35 - 0.00005 x 2061 = 0.2470.
    assert r["gear"] == 5 and r["rpm"] == pytest.approx(2807, abs=1)
    assert r["efficiency"] == pytest.approx(0.3347, abs=0.0005)
    assert r["actual"]["gear"] == 3 and r["actual"]["efficiency"] == pytest.approx(0.2470, abs=0.0005)
    assert r["efficiency_gain"] == pytest.approx(0.0877, abs=0.001)


def test_fuel_saved_follows_from_the_engine_power_and_the_two_efficiencies():
    # Engine power 15054 / 0.9 = 16727 W. Fuel power = that / efficiency: 49982 W (best) and 67734 W (3rd). One litre of
    # petrol holds 0.745 kg x 43.4 MJ/kg = 32.33 MJ, so 49982 W burns 5.565 L/h and 67734 W burns 7.541 L/h.
    r = optimise(100, P100, PeakAt2500(), SPEC, actual_gear=3)

    assert r["fuel_l_per_h"] == pytest.approx(5.565, rel=0.01) and r["actual"]["fuel_l_per_h"] == pytest.approx(7.541, rel=0.01)
    assert r["fuel_saved_l_per_h"] == pytest.approx(1.976, rel=0.02)
    assert r["rupees_saved_per_h"] == pytest.approx(1.976 * 105, rel=0.02)
    assert optimise(100, P100, PeakAt2500(), SPEC, actual_gear=3, price_per_litre=200)["rupees_saved_per_h"] == pytest.approx(1.976 * 200, rel=0.02)


def test_a_misfire_scales_efficiency_by_its_assumed_loss_and_reports_what_the_fault_costs():
    healthy = optimise(100, P100, PeakAt2500(), SPEC, actual_gear=3)
    r = optimise(100, P100, PeakAt2500(), SPEC, fault="misfire", severity=0.8, actual_gear=3)

    # loss = 0.25 x 0.8 = 20% of the healthy efficiency, so fuel use rises by 1 / 0.8 - 1 = 25%
    assert r["efficiency"] == pytest.approx(0.8 * healthy["efficiency"])
    assert r["efficiency_lost_to_fault"] == pytest.approx(0.2 * healthy["efficiency"])
    assert r["fault_loss_fraction"] == pytest.approx(0.2)
    assert r["fault_extra_fuel_l_per_h"] == pytest.approx(0.25 * healthy["fuel_l_per_h"], rel=0.001)


def test_loss_rates_from_the_sliders_change_the_fault_cost():
    r = optimise(100, P100, PeakAt2500(), SPEC, fault="misfire", severity=1.0, loss_rates={"misfire": 0.5})

    assert r["fault_loss_fraction"] == pytest.approx(0.5)


def test_a_healthy_engine_loses_nothing_to_faults():
    r = optimise(100, P100, PeakAt2500(), SPEC)

    assert r["efficiency_lost_to_fault"] == 0.0 and r["fault_extra_fuel_l_per_h"] == 0.0


def test_knock_makes_the_optimiser_stay_clear_of_heavy_load_at_low_speed():
    # At 50 km/h with 16.6 kW at the wheels a model that likes 1700 rpm picks 4th gear (80% of full load) when healthy.
    # With full-severity knock the limit falls to 0.85 x (1 - 0.3) = 0.595 of full load, so 4th is out and it takes 3rd.
    model = PeakAt2500(peak=1700.0)

    assert optimise(50, 16555.0, model, SPEC)["gear"] == 4
    assert optimise(50, 16555.0, model, SPEC, fault="knock", severity=1.0)["gear"] == 3


def test_no_power_demand_means_no_fuel_and_no_gain():
    r = optimise(60, 0.0, PeakAt2500(), SPEC)

    assert r["fuel_l_per_h"] == 0.0 and r["fuel_saved_l_per_h"] == 0.0 and r["efficiency_gain"] == 0.0


def test_when_no_gear_fits_the_limits_the_lowest_gear_is_returned_flagged_as_forced():
    r = optimise(4, wheel_power_w(4, 0.5, SPEC), PeakAt2500(), SPEC)  # 4 km/h is under idle in every gear

    assert r["gear"] == 1 and r["forced"] is True
