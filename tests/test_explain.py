import numpy as np

from enginetwin.explain import healthy_baseline, reasons
from enginetwin.fault_features import FEATURE_NAMES

NORMAL = {
    "sound_half_order_ratio": 0.05, "vibration_half_order_ratio": 0.05, "vibration_cyl_min_ratio": 0.9,
    "obd_total_fuel_trim": 4.0, "vibration_knock_vs_floor": 0.0, "sound_knock_vs_floor": 0.0,
    "vibration_knock_env_firing": 2.0, "obd_timing_advance": 30.0, "vibration_valve_vs_floor": 0.0,
    "sound_valve_vs_floor": 0.0, "vibration_valve_env_cycle": 2.0,
}


def vector(**changes):
    values = {name: 1.0 for name in FEATURE_NAMES}
    values.update(NORMAL)
    values.update(changes)
    return np.array([values[n] for n in FEATURE_NAMES])


def baseline():
    """Healthy rows that wobble by 0.1% around the normal values, so a real change is many standard deviations."""
    rng = np.random.default_rng(0)
    return healthy_baseline(vector() * (1 + 0.001 * rng.standard_normal((200, len(FEATURE_NAMES)))))


def test_a_misfire_is_explained_in_the_wording_of_the_brief():
    # half-order vibration 0.40 against a normal 0.05 is 8 times; fuel trim 10 against 4 is up 6
    r = reasons(vector(vibration_half_order_ratio=0.40, obd_total_fuel_trim=10.0), "misfire", baseline())

    assert "half-order vibration 8 times normal" in r and "fuel trim up 6%" in r


def test_the_strongest_change_comes_first_and_top_limits_the_list():
    v = vector(vibration_half_order_ratio=0.40, sound_half_order_ratio=0.10, obd_total_fuel_trim=10.0)
    r = reasons(v, "misfire", baseline(), top=2)

    assert len(r) == 2 and r[0].startswith("half-order vibration")  # a 7x jump beats a 2x one


def test_a_dead_cylinder_is_described_by_the_share_of_the_average_firing_energy_it_carries():
    r = reasons(vector(vibration_cyl_min_ratio=0.1), "misfire", baseline())

    assert "weakest cylinder carries only 10% of the average firing energy (normal 90%)" in r


def test_knock_is_explained_by_high_frequency_energy_and_retarded_timing():
    # the floor-relative feature is a log10 ratio: 1.0 against a normal 0.0 is 10 times the background
    r = reasons(vector(vibration_knock_vs_floor=1.0, obd_timing_advance=24.0), "knock", baseline())

    assert "vibration energy at 6-7 kHz 10 times the background" in r and "ignition timing pulled back 6 degrees" in r


def test_valve_wear_is_explained_by_the_tick_band_and_rhythm():
    r = reasons(vector(vibration_valve_vs_floor=1.0, vibration_valve_env_cycle=10.0), "valve_wear", baseline())

    assert "vibration ticks at 2.5-3.5 kHz 10 times the background" in r and "valve tick rhythm once per two revolutions 5 times normal" in r


def test_a_combined_label_is_explained_by_both_faults():
    r = reasons(vector(vibration_half_order_ratio=0.40, vibration_knock_vs_floor=1.0), "misfire+knock", baseline())

    assert any("half-order" in s for s in r) and any("6-7 kHz" in s for s in r)


def test_readings_that_have_not_moved_are_not_cited_and_a_healthy_label_says_so():
    assert reasons(vector(), "healthy", baseline()) == ["all readings are within their normal range"]
    assert reasons(vector(), "misfire", baseline()) == ["no single reading stands out strongly"]


def test_a_reason_that_does_not_name_a_channel_is_given_once_even_if_both_channels_show_it():
    r = reasons(vector(vibration_cyl_min_ratio=0.1, sound_cyl_min_ratio=0.1), "misfire", baseline(), top=5)

    assert sum("weakest cylinder" in s for s in r) == 1
