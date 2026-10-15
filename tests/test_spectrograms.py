import numpy as np
import pytest

from enginetwin.schema import Window
from enginetwin.spectrograms import spectrogram_stack, standardise_spectrogram, window_spectrogram
from enginetwin.labelled import make_labelled_set

FS = 16000
OBD = {"engine_speed": 2000.0}


def window(sound, vibration=None):
    return Window(
        sound=sound, sound_fs=FS, vibration=sound if vibration is None else vibration, vibration_fs=FS,
        obd=OBD, fault="healthy", severity=0.0, source="real",
    )


def sine(hz, amplitude=1.0, seconds=2.0):
    return amplitude * np.sin(2 * np.pi * hz * np.arange(int(seconds * FS)) / FS)


def test_a_window_becomes_two_64_by_64_log_spectrograms_sound_first_then_vibration():
    spec = window_spectrogram(window(sine(1000), vibration=sine(3000)))

    assert spec.shape == (2, 64, 64)  # (channel, frequency band, time frame)
    assert spec[0].mean(axis=1).argmax() == 8  # 1000 Hz sits in the 125 Hz-wide band 8 (bands 0..63 cover 0 to 8 kHz)
    assert spec[1].mean(axis=1).argmax() == 24  # 3000 / 125


def test_an_event_shows_up_only_in_the_frames_where_it_happens():
    sound = np.concatenate([np.zeros(FS), sine(2000, seconds=1.0)])  # silent first second, tone in the second
    spec = window_spectrogram(window(sound))[0]
    band = spec[16]  # 2000 / 125

    assert band[:28].max() < band[36:].min() - 2  # first half is at the floor, second half is >2 log10 units higher


def test_ten_times_the_amplitude_adds_one_to_the_log10_spectrogram():
    quiet = window_spectrogram(window(sine(1000, 0.1)))[0, 8].mean()
    loud = window_spectrogram(window(sine(1000, 1.0)))[0, 8].mean()

    assert loud - quiet == pytest.approx(1.0, abs=0.02)


def test_standardising_gives_every_channel_zero_mean_and_unit_spread():
    spec = standardise_spectrogram(window_spectrogram(window(sine(1000) + 0.1 * np.random.default_rng(0).normal(size=2 * FS))))

    assert np.allclose(spec.mean(axis=(1, 2)), 0.0, atol=1e-6) and np.allclose(spec.std(axis=(1, 2)), 1.0, atol=1e-6)


def test_standardised_spectrograms_ignore_the_overall_signal_level():
    # Scaling a recording by 10 adds exactly 1 to every log10 bin, and standardising removes a constant shift.
    noisy = sine(1000) + 0.1 * np.random.default_rng(0).normal(size=2 * FS)
    a = standardise_spectrogram(window_spectrogram(window(noisy)))
    b = standardise_spectrogram(window_spectrogram(window(10 * noisy)))

    assert np.abs(a - b).mean() < 0.005 and np.abs(a - b).max() < 0.06  # exact apart from the 1e-5 floor added before the log, felt only in the quietest bins


def test_a_silent_recording_standardises_to_zeros_not_nan():
    assert np.array_equal(standardise_spectrogram(window_spectrogram(window(np.zeros(2 * FS)))), np.zeros((2, 64, 64)))


def small_set():
    return make_labelled_set(2, rpm_range=(800, 3000), snr_db_range=(15, 25), seed=1, healthy_efficiency=lambda r, l: 0.3)


def test_the_stack_has_one_spectrogram_per_window_equal_to_that_windows_own():
    s = small_set()
    stack = spectrogram_stack(s)

    assert stack.shape == (len(s), 2, 64, 64) and stack.dtype == np.float16
    assert np.allclose(stack[3].astype(float), standardise_spectrogram(window_spectrogram(s.window(3))), atol=0.01)


def test_the_stack_is_built_once_and_read_back_from_its_cache_file(tmp_path, monkeypatch):
    import enginetwin.spectrograms as sp

    s = small_set()
    path = tmp_path / "spec.npy"
    first = sp.spectrogram_stack(s, cache_path=path)
    monkeypatch.setattr(sp, "window_spectrogram", lambda w: pytest.fail("recomputed"))

    assert np.array_equal(first, sp.spectrogram_stack(s, cache_path=path))
