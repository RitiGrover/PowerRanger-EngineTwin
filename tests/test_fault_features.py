import numpy as np
import pytest

from enginetwin.fault_features import FEATURE_NAMES, extract_features, refine_rpm
from enginetwin.schema import FAULTS, Window
from enginetwin.simulator import simulate_window

FS = 16000
OBD = {
    "engine_speed": 2000.0, "calculated_load_value": 40.0, "engine_coolant_temperature": 85.0, "maf": 10.0,
    "timing_advance": 30.0, "short_term_fuel_trim_bank1": 1.5, "long_term_fuel_trim_bank1": 2.5,
}


def feature(name, w):
    vector, names = extract_features(w)
    return vector[names.index(name)]


def mean_feature(name, n=5, **kwargs):
    """Average over several seeds so the check is about the fault, not one noisy window."""
    return np.mean([feature(name, simulate_window(seed=s, **kwargs)) for s in range(n)])


def synthetic(sound, obd=OBD):
    return Window(sound=sound, sound_fs=FS, vibration=sound, vibration_fs=FS, obd=obd, fault="healthy", severity=0.0, source="real")


def test_there_are_about_forty_uniquely_named_finite_features_for_every_fault_label():
    assert 38 <= len(FEATURE_NAMES) <= 45 and len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    for fault in FAULTS:
        vector, names = extract_features(simulate_window(rpm=2400, fault=fault, severity=0.8, seed=1))

        assert names == FEATURE_NAMES and vector.shape == (len(FEATURE_NAMES),)
        assert np.all(np.isfinite(vector))


def test_the_same_window_always_gives_the_same_features():
    w = simulate_window(rpm=2400, fault="knock", severity=0.7, seed=3)

    assert np.array_equal(extract_features(w)[0], extract_features(w)[0])


def test_a_silent_recording_gives_finite_features_instead_of_nan():
    vector, _ = extract_features(synthetic(np.zeros(FS)))

    assert np.all(np.isfinite(vector))


def test_the_obd_features_are_the_readings_and_their_total_fuel_trim():
    assert feature("obd_engine_speed", synthetic(np.zeros(FS))) == 2000.0
    assert feature("obd_timing_advance", synthetic(np.zeros(FS))) == 30.0
    assert feature("obd_total_fuel_trim", synthetic(np.zeros(FS))) == pytest.approx(4.0)  # 1.5 + 2.5


# Time and spectrum statistics, checked on a pure 1 kHz sine of amplitude 1, where the answers are known by hand.
@pytest.fixture
def sine():
    t = np.arange(2 * FS) / FS
    return synthetic(np.sin(2 * np.pi * 1000 * t))


def test_rms_is_amplitude_over_root_two_and_crest_factor_is_root_two_for_a_sine(sine):
    assert feature("sound_rms", sine) == pytest.approx(1 / np.sqrt(2), rel=0.01)
    assert feature("sound_crest", sine) == pytest.approx(np.sqrt(2), rel=0.01)


def test_kurtosis_is_1_5_for_a_sine_and_3_for_gaussian_noise(sine):
    # sine: E[sin^4] / E[sin^2]^2 = (3/8) / (1/4) = 1.5
    noise = synthetic(np.random.default_rng(0).normal(size=2 * FS))

    assert feature("sound_kurtosis", sine) == pytest.approx(1.5, rel=0.01)
    assert feature("sound_kurtosis", noise) == pytest.approx(3.0, rel=0.05)


def test_the_spectral_centroid_and_rolloff_of_a_1_khz_sine_sit_at_1_khz_and_it_is_not_flat(sine):
    assert feature("sound_centroid_hz", sine) == pytest.approx(1000, rel=0.01)
    assert feature("sound_rolloff_hz", sine) == pytest.approx(1000, abs=10)
    assert feature("sound_flatness", sine) < 0.01


def test_white_noise_is_spectrally_flat_and_a_sine_is_not():
    noise = synthetic(np.random.default_rng(1).normal(size=2 * FS))

    assert feature("sound_flatness", noise) > 0.5  # geometric/arithmetic mean of a chi-square-like spectrum is ~0.56
    assert feature("sound_centroid_hz", noise) == pytest.approx(4000, rel=0.05)  # flat up to the 8 kHz Nyquist


def test_a_misfire_raises_the_half_order_ratio_in_sound_and_vibration():
    for channel in ("sound", "vibration"):
        healthy = mean_feature(f"{channel}_half_order_ratio", rpm=2400)
        misfire = mean_feature(f"{channel}_half_order_ratio", rpm=2400, fault="misfire", severity=1.0)

        assert healthy < 0.1 and misfire > 0.2  # the same figures the simulator tests derive from an impulse train


def test_knock_raises_the_knock_band_fraction_far_above_healthy():
    healthy = mean_feature("vibration_knock_band_fraction", rpm=2400)
    knock = mean_feature("vibration_knock_band_fraction", rpm=2400, fault="knock", severity=1.0)

    assert knock > 10 * healthy


def test_valve_wear_raises_the_valve_band_fraction_and_the_once_per_cycle_envelope_line_but_not_the_knock_band():
    healthy = dict(rpm=2400)
    wear = dict(rpm=2400, fault="valve_wear", severity=1.0)

    assert mean_feature("vibration_valve_band_fraction", **wear) > 10 * mean_feature("vibration_valve_band_fraction", **healthy)
    assert mean_feature("vibration_valve_env_cycle", **wear) > 3 * mean_feature("vibration_valve_env_cycle", **healthy)
    assert mean_feature("vibration_knock_band_fraction", **wear) < 3 * mean_feature("vibration_knock_band_fraction", **healthy)


def test_knock_bursts_after_every_firing_show_as_an_envelope_line_at_the_firing_rate():
    healthy = mean_feature("vibration_knock_env_firing", rpm=2400)
    knock = mean_feature("vibration_knock_env_firing", rpm=2400, fault="knock", severity=1.0)

    assert knock > 3 * healthy


def test_a_dead_cylinder_makes_the_cylinder_energy_very_uneven_and_a_healthy_engine_even():
    healthy = mean_feature("vibration_cyl_min_ratio", rpm=2400)
    misfire = mean_feature("vibration_cyl_min_ratio", rpm=2400, fault="misfire", severity=1.0)

    assert healthy > 0.7  # cylinders differ by a few percent
    assert misfire < 0.2  # one of four carries only noise: min / mean is about 0
    assert mean_feature("vibration_cyl_cv", rpm=2400, fault="misfire", severity=1.0) > 3 * mean_feature("vibration_cyl_cv", rpm=2400)


def test_the_firing_rate_is_found_in_the_sound_even_when_the_obd_speed_is_off_by_8_percent():
    # 2400 rpm fires 80 times a second, so the sound's strongest line near the OBD figure gives the true speed.
    w = simulate_window(rpm=2400, seed=2)

    assert refine_rpm(w.sound, w.sound_fs, rpm_obd=2400 * 0.92) == pytest.approx(2400, rel=0.003)
    assert refine_rpm(w.sound, w.sound_fs, rpm_obd=2400 * 1.08) == pytest.approx(2400, rel=0.003)


def test_cylinder_features_still_find_a_dead_cylinder_when_the_obd_speed_has_dipped():
    # Misfire dips the OBD rpm by about 90 rpm, which would drift the cylinder windows if it were used unrefined.
    w = simulate_window(rpm=1200, fault="misfire", severity=1.0, seed=4, cylinder=3)
    w.obd["engine_speed"] = 1200 - 90

    assert feature("vibration_cyl_min_ratio", w) < 0.2


# feature_matrix: one row per record of a labelled set, cached to disk so training never re-extracts.
def tiny_set():
    from enginetwin.labelled import make_labelled_set

    return make_labelled_set(2, rpm_range=(800, 3000), snr_db_range=(15, 25), seed=1, healthy_efficiency=lambda r, l: 0.3)


def test_feature_matrix_has_one_row_per_window_equal_to_that_windows_features():
    from enginetwin.fault_features import feature_matrix

    s = tiny_set()
    X = feature_matrix(s)

    assert X.shape == (len(s), len(FEATURE_NAMES))
    for i in (0, 5, len(s) - 1):
        assert np.allclose(X[i], extract_features(s.window(i))[0])


def test_feature_matrix_is_computed_once_and_read_back_from_the_cache_file(tmp_path, monkeypatch):
    import enginetwin.fault_features as ff

    s = tiny_set()
    path = tmp_path / "train.npy"
    first = ff.feature_matrix(s, cache_path=path)
    assert path.exists()

    monkeypatch.setattr(ff, "extract_features", lambda w: pytest.fail("features were recomputed"))
    again = ff.feature_matrix(s, cache_path=path)

    assert np.array_equal(first, again)


def test_a_cache_file_of_the_wrong_length_is_not_trusted(tmp_path):
    from enginetwin.fault_features import feature_matrix

    s = tiny_set()
    path = tmp_path / "x.npy"
    np.save(path, np.zeros((3, len(FEATURE_NAMES))))

    assert feature_matrix(s, cache_path=path).shape == (len(s), len(FEATURE_NAMES))


# Floor-relative band features: log10 of the band's power over the power in nearby bands, so a rise in broadband noise
# moves band and floor together and the feature stays put.
@pytest.mark.parametrize("sigma", [0.01, 1.0])
def test_white_noise_gives_a_floor_relative_feature_near_zero_whatever_its_level(sigma):
    w = synthetic(np.random.default_rng(3).normal(0, sigma, 2 * FS))

    for name in ("sound_knock_vs_floor", "sound_valve_vs_floor"):
        assert feature(name, w) == pytest.approx(0.0, abs=0.1)  # equal power in band and floor: log10(1)


def test_a_tone_in_the_knock_band_raises_its_floor_relative_feature_by_the_hand_calculated_amount():
    # Noise sigma 0.1 has mean square 0.01 spread evenly over 0-8 kHz, so 0.00125 falls in the 1 kHz knock band. A 6.5 kHz
    # sine of amplitude 1 adds a mean square of 0.5 there: the band power rises by (0.5 + 0.00125) / 0.00125 = 401 times
    # while the floor is untouched, so the feature rises by log10(401) = 2.60.
    t = np.arange(2 * FS) / FS
    noise = np.random.default_rng(3).normal(0, 0.1, 2 * FS)
    quiet, tone = synthetic(noise), synthetic(noise + np.sin(2 * np.pi * 6500 * t))

    assert feature("sound_knock_vs_floor", tone) - feature("sound_knock_vs_floor", quiet) == pytest.approx(2.60, abs=0.1)
    assert feature("sound_valve_vs_floor", tone) == pytest.approx(feature("sound_valve_vs_floor", quiet), abs=0.1)


def test_the_healthy_baseline_of_the_knock_feature_stays_at_zero_as_noise_rises_and_knock_still_stands_out():
    # The plain band fraction of a healthy window climbs with noise (it dilutes the knock/healthy gap); this one does not.
    healthy = {snr: mean_feature("vibration_knock_vs_floor", rpm=2400, snr_db=snr) for snr in (20.0, 5.0)}
    knock_10 = mean_feature("vibration_knock_vs_floor", rpm=2400, fault="knock", severity=1.0, snr_db=10.0)

    assert all(abs(v) < 0.1 for v in healthy.values())
    assert knock_10 > 0.3


def test_valve_wear_stands_out_from_its_floor_at_10_db_snr():
    healthy = mean_feature("vibration_valve_vs_floor", rpm=2400, snr_db=10.0)
    wear = mean_feature("vibration_valve_vs_floor", rpm=2400, fault="valve_wear", severity=1.0, snr_db=10.0)

    assert wear > healthy + 0.5


def test_the_misfire_ratio_survives_5_db_of_noise():
    # Ratio features should not need replacing: order 0.5 over order 2 keeps its separation at low SNR.
    healthy = mean_feature("vibration_half_order_ratio", rpm=2400, snr_db=5.0)
    misfire = mean_feature("vibration_half_order_ratio", rpm=2400, fault="misfire", severity=1.0, snr_db=5.0)

    assert misfire > 3 * healthy and misfire > 0.2
