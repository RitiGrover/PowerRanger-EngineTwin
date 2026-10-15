import numpy as np
import pytest

from enginetwin.noise import estimate_noise_db, fit_noise_thresholds, noise_bins
from enginetwin.schema import Window
from enginetwin.simulator import simulate_window

FS = 16000
OBD = {"engine_speed": 2000.0}


def window(sound):
    return Window(sound=sound, sound_fs=FS, vibration=sound, vibration_fs=FS, obd=OBD, fault="healthy", severity=0.0, source="real")


def tone_plus_noise(sigma, seed=0):
    t = np.arange(2 * FS) / FS
    return np.sin(2 * np.pi * 1000 * t) + np.random.default_rng(seed).normal(0, sigma, 2 * FS)


def test_the_noise_level_is_the_noise_to_signal_ratio_in_db_read_from_the_7_5_to_8_khz_floor():
    # A 1 kHz sine of amplitude 1 has mean square 0.5; white noise of sigma 0.1 adds 0.01 everywhere, including the
    # 7.5-8 kHz band where the tone is absent. Noise / signal = 0.01 / 0.5 = 1/50: -17.0 dB (an SNR of 17 dB).
    assert estimate_noise_db(window(tone_plus_noise(0.1))) == pytest.approx(-17.0, abs=0.5)


def test_ten_times_the_noise_amplitude_adds_20_db():
    quiet = estimate_noise_db(window(tone_plus_noise(0.05)))
    loud = estimate_noise_db(window(tone_plus_noise(0.5)))

    assert loud - quiet == pytest.approx(20.0, abs=1.0)


@pytest.mark.parametrize("snr_db", [20.0, 10.0, 5.0, 0.0])
@pytest.mark.parametrize("rpm", [1200.0, 2400.0, 4000.0])
def test_simulated_windows_read_back_close_to_minus_their_snr_at_any_speed_and_for_a_fault_too(snr_db, rpm):
    for fault, severity in (("healthy", 0.0), ("misfire", 0.8)):
        w = simulate_window(rpm=rpm, fault=fault, severity=severity, snr_db=snr_db, seed=3)

        assert estimate_noise_db(w) == pytest.approx(-snr_db, abs=2.5)


def test_a_silent_recording_does_not_give_nan():
    assert np.isfinite(estimate_noise_db(window(np.zeros(2 * FS))))


EDGES = (-20.0, -15.0, -10.0, -5.0)  # five bins: below -20, then 5 dB steps, then above -5


def test_each_noise_level_falls_in_the_bin_between_its_two_edges():
    assert noise_bins(np.array([-30.0, -20.0, -17.0, -12.0, -7.0, -5.0, 0.0]), EDGES).tolist() == [0, 1, 1, 2, 3, 4, 4]


def scores_and_noise(seed=0):
    """Healthy reconstruction error that grows 10x per 10 dB of noise, 200 windows in each of three noise levels."""
    rng = np.random.default_rng(seed)
    noise = np.repeat([-22.0, -12.0, -3.0], 200) + rng.uniform(-1, 1, 600)
    scores = rng.gamma(4.0, 0.25, 600) * 10 ** ((noise + 22) / 10)
    return scores, noise


def test_each_bin_gets_its_own_alarm_level_from_the_healthy_windows_in_it():
    scores, noise = scores_and_noise()
    labels = np.array(["healthy"] * 600)
    thr = fit_noise_thresholds(scores, noise, labels, EDGES, q=0.99)

    for level, in_bin in ((-22.0, noise < -20), (-12.0, (noise >= -15) & (noise < -10)), (-3.0, noise >= -5)):
        assert thr.threshold_for(level) == pytest.approx(np.quantile(scores[in_bin], 0.99))
    assert thr.threshold_for(-3.0) > 50 * thr.threshold_for(-22.0)  # 19 dB apart


def test_faulty_windows_do_not_count_towards_an_alarm_level():
    scores, noise = scores_and_noise()
    clean = fit_noise_thresholds(scores, noise, np.array(["healthy"] * 600), EDGES)
    mixed = fit_noise_thresholds(scores, noise, np.array(["healthy"] * 300 + ["misfire"] * 300), EDGES)

    assert mixed.threshold_for(-22.0) == clean.threshold_for(-22.0)  # the first 300 rows are the two quietest levels
    assert mixed.threshold_for(-3.0) == mixed.threshold_for(-12.0)  # no healthy window is left in the noisiest bin: borrowed


def test_a_noise_level_with_no_healthy_calibration_windows_borrows_the_nearest_bin_and_the_noisier_one_on_a_tie():
    scores, noise = scores_and_noise()
    thr = fit_noise_thresholds(scores, noise, np.array(["healthy"] * 600), EDGES)

    assert thr.threshold_for(-17.0) == thr.threshold_for(-12.0)  # bin 1 sits between bins 0 and 2: take the noisier
    assert thr.threshold_for(-7.0) == thr.threshold_for(-3.0)  # bin 3 is next to the populated bin 4 and one from bin 2: take 4
    assert thr.threshold_for(10.0) == thr.threshold_for(-3.0)  # beyond the noisiest calibrated bin


def test_threshold_for_accepts_arrays_and_flags_are_per_window():
    scores, noise = scores_and_noise()
    thr = fit_noise_thresholds(scores, noise, np.array(["healthy"] * 600), EDGES)

    assert thr.threshold_for(noise).shape == (600,)
    assert np.mean(scores > thr.threshold_for(noise)) < 0.03  # about 1% of the healthy windows it was built from
