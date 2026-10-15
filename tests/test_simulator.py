import numpy as np
import pytest

from enginetwin.signals import (
    band_energy_fraction,
    bandpass,
    envelope_spectrum,
    half_order_ratio,
    order_amplitude,
    per_cylinder_energy,
)
from enginetwin.simulator import simulate_window

FS = 16000


def test_a_healthy_window_is_two_seconds_of_labelled_simulated_data():
    w = simulate_window(rpm=2000, seed=1)

    assert w.duration_s == 2.0
    assert len(w.vibration) == 2 * w.vibration_fs
    assert (w.fault, w.severity, w.source) == ("healthy", 0.0, "simulated")


@pytest.mark.parametrize("rpm", [1500, 3000])
def test_the_dominant_low_order_line_is_the_firing_frequency_at_any_rpm(rpm):
    w = simulate_window(rpm=rpm, seed=3)

    orders = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    strengths = [order_amplitude(w.sound, w.sound_fs, rpm, o) for o in orders]

    assert orders[int(np.argmax(strengths))] == 2.0  # two firings per revolution


def _half(w, rpm):
    return half_order_ratio(w.sound, w.sound_fs, rpm)


def test_a_misfire_creates_half_order_content_that_grows_with_severity():
    # One firing missing out of every four leaves a pattern repeating once per cycle (order 0.5). For a clean
    # impulse train [1,1,0,1] the line at order 0.5 is 1/4 and the firing line is 3/4: a ratio of about 1/3.
    rpm = 2400
    healthy = _half(simulate_window(rpm=rpm, seed=5), rpm)
    half = _half(simulate_window(rpm=rpm, fault="misfire", severity=0.5, seed=5), rpm)
    full = _half(simulate_window(rpm=rpm, fault="misfire", severity=1.0, seed=5), rpm)

    assert healthy < 0.1
    assert healthy < half < full
    assert 0.2 <= full <= 0.5


@pytest.mark.parametrize("weak", [1, 2, 3, 4])
def test_the_misfiring_cylinder_is_the_one_with_the_least_firing_energy(weak):
    rpm = 1200
    w = simulate_window(rpm=rpm, fault="misfire", severity=1.0, cylinder=weak, seed=11)

    energy = per_cylinder_energy(w.vibration, w.vibration_fs, rpm)

    assert min(energy, key=energy.get) == weak
    others = [e for c, e in energy.items() if c != weak]
    assert energy[weak] < 0.2 * min(others)  # a dead cylinder carries only noise


def test_knock_adds_6_to_7_khz_energy_that_grows_with_severity():
    rpm = 2400
    shares = []
    for fault, severity in (("healthy", 0.0), ("knock", 0.5), ("knock", 1.0)):
        w = simulate_window(rpm=rpm, fault=fault, severity=severity, seed=9)
        shares.append(band_energy_fraction(w.sound, w.sound_fs, 6000, 7000))

    healthy, half, full = shares
    assert healthy < half < full
    assert full > 10 * healthy


def test_knock_bursts_start_just_after_each_firing():
    rpm = 2400
    w = simulate_window(rpm=rpm, fault="knock", severity=1.0, seed=9)
    knock_band = bandpass(w.sound, w.sound_fs, 6000, 7000) ** 2
    interval = 30.0 / rpm

    in_window = 0.0
    for k in range(int(w.duration_s / interval) - 1):
        start = int(k * interval * w.sound_fs)
        in_window += knock_band[start : start + int(0.003 * w.sound_fs)].sum()  # the 3 ms after the firing

    assert in_window / knock_band.sum() > 0.8  # steady noise would put only 3/12.5 = 24% there


def mean_obd(key, **kwargs):
    """Average of one OBD reading over 200 seeds. This keeps the check about the fault and not one noisy sample."""
    return np.mean([simulate_window(seed=s, duration_s=0.1, **kwargs).obd[key] for s in range(200)])


def test_obd_readings_use_the_same_names_as_the_real_data_loaders():
    obd = simulate_window(rpm=2000, seed=1).obd

    assert {
        "engine_speed", "calculated_load_value", "engine_coolant_temperature", "maf", "timing_advance",
        "short_term_fuel_trim_bank1", "long_term_fuel_trim_bank1",
    } <= set(obd)


def test_a_misfire_dips_rpm_and_raises_fuel_trim_but_leaves_timing_alone():
    healthy = dict(rpm=2000)
    misfire = dict(rpm=2000, fault="misfire", severity=1.0)

    assert mean_obd("engine_speed", **misfire) < mean_obd("engine_speed", **healthy) - 20
    assert mean_obd("short_term_fuel_trim_bank1", **misfire) > mean_obd("short_term_fuel_trim_bank1", **healthy) + 3
    assert abs(mean_obd("timing_advance", **misfire) - mean_obd("timing_advance", **healthy)) < 1


def test_knock_retards_the_timing_but_leaves_fuel_trim_alone():
    healthy = dict(rpm=2000)
    knock = dict(rpm=2000, fault="knock", severity=1.0)

    assert mean_obd("timing_advance", **knock) < mean_obd("timing_advance", **healthy) - 3
    assert abs(mean_obd("short_term_fuel_trim_bank1", **knock) - mean_obd("short_term_fuel_trim_bank1", **healthy)) < 1


def test_the_same_seed_gives_the_same_signal_and_a_different_seed_does_not():
    a = simulate_window(rpm=2000, seed=7)
    b = simulate_window(rpm=2000, seed=7)
    c = simulate_window(rpm=2000, seed=8)

    assert np.array_equal(a.sound, b.sound)
    assert not np.array_equal(a.sound, c.sound)


def quiet_gaps(signal, fs, rpm, start_s=0.03, stop_s=0.09):
    """Samples between combustion pulses at a very low rpm. Each pulse has died out by then (decay < 0.1% by 30 ms)."""
    interval = 30.0 / rpm
    pieces = [signal[int((k * interval + start_s) * fs) : int((k * interval + stop_s) * fs)] for k in range(int(2 / interval) - 1)]
    return np.concatenate(pieces)


def test_noise_sigma_sets_the_background_noise_level():
    rpm = 300  # one firing every 100 ms
    quiet = simulate_window(rpm=rpm, seed=2, noise_sigma=0.05)
    silent = simulate_window(rpm=rpm, seed=2, noise_sigma=0.0)

    assert np.std(quiet_gaps(quiet.sound, quiet.sound_fs, rpm)) == pytest.approx(0.05, rel=0.05)
    assert np.std(quiet_gaps(quiet.vibration, quiet.vibration_fs, rpm)) == pytest.approx(0.05, rel=0.05)
    assert np.max(np.abs(quiet_gaps(silent.sound, silent.sound_fs, rpm))) < 0.001  # only the pulse tails remain


def valve_share(w):
    return band_energy_fraction(w.vibration, w.vibration_fs, 2500, 3500)


def test_valve_wear_adds_2500_to_3500_hz_energy_that_grows_with_severity():
    shares = [
        valve_share(simulate_window(rpm=2400, fault=fault, severity=severity, seed=4))
        for fault, severity in (("healthy", 0.0), ("valve_wear", 0.5), ("valve_wear", 1.0))
    ]

    healthy, half, full = shares
    assert healthy < half < full
    assert full > 10 * healthy


def envelope_line(w, hz):
    freqs, amplitudes = envelope_spectrum(w.vibration, w.vibration_fs, band=(2500, 3500))
    return amplitudes[np.abs(freqs - hz) <= 1.0].max()


def test_valve_ticks_repeat_at_the_firing_rate():
    rpm = 2400  # a valve event per firing: 2400 / 30 = 80 Hz; nothing at half that rate
    healthy = simulate_window(rpm=rpm, seed=4)
    wear = simulate_window(rpm=rpm, fault="valve_wear", severity=1.0, seed=4)

    assert envelope_line(wear, rpm / 30) > 3 * envelope_line(healthy, rpm / 30)
    assert envelope_line(wear, rpm / 60) < 0.1 * envelope_line(wear, rpm / 30)


def test_valve_wear_is_not_mistaken_for_knock():
    knock_share = lambda w: band_energy_fraction(w.vibration, w.vibration_fs, 6000, 7000)
    wear = simulate_window(rpm=2400, fault="valve_wear", severity=1.0, seed=4)
    knock = simulate_window(rpm=2400, fault="knock", severity=1.0, seed=4)

    assert knock_share(wear) < 0.1 * knock_share(knock)


def test_valve_wear_is_mechanical_so_obd_readings_do_not_move():
    wear = dict(rpm=2000, fault="valve_wear", severity=1.0)

    assert abs(mean_obd("engine_speed", **wear) - mean_obd("engine_speed", rpm=2000)) < 5
    assert abs(mean_obd("timing_advance", **wear) - mean_obd("timing_advance", rpm=2000)) < 1
    assert abs(mean_obd("short_term_fuel_trim_bank1", **wear) - mean_obd("short_term_fuel_trim_bank1", rpm=2000)) < 1


# Combined faults: label "a+b" applies both faults at the one severity, each leaving its own fingerprint.
def test_misfire_plus_knock_shows_both_fingerprints():
    rpm = 2400
    healthy = simulate_window(rpm=rpm, seed=5)
    w = simulate_window(rpm=rpm, fault="misfire+knock", severity=1.0, seed=5)

    assert _half(w, rpm) > 0.2  # misfire: order-0.5 line
    assert band_energy_fraction(w.sound, w.sound_fs, 6000, 7000) > 10 * band_energy_fraction(healthy.sound, healthy.sound_fs, 6000, 7000)


def test_misfire_plus_valve_wear_shows_both_fingerprints():
    rpm = 2400
    healthy = simulate_window(rpm=rpm, seed=5)
    w = simulate_window(rpm=rpm, fault="misfire+valve_wear", severity=1.0, seed=5)

    assert _half(w, rpm) > 0.2
    assert valve_share(w) > 10 * valve_share(healthy)


def test_knock_plus_valve_wear_shows_both_fingerprints():
    rpm = 2400
    healthy = simulate_window(rpm=rpm, seed=5)
    w = simulate_window(rpm=rpm, fault="knock+valve_wear", severity=1.0, seed=5)

    assert band_energy_fraction(w.vibration, w.vibration_fs, 6000, 7000) > 10 * band_energy_fraction(healthy.vibration, healthy.vibration_fs, 6000, 7000)
    assert valve_share(w) > 10 * valve_share(healthy)


def test_combined_faults_combine_their_obd_effects():
    both = dict(rpm=2000, fault="misfire+knock", severity=1.0)

    assert mean_obd("engine_speed", **both) < mean_obd("engine_speed", rpm=2000) - 20
    assert mean_obd("short_term_fuel_trim_bank1", **both) > mean_obd("short_term_fuel_trim_bank1", rpm=2000) + 3
    assert mean_obd("timing_advance", **both) < mean_obd("timing_advance", rpm=2000) - 3
