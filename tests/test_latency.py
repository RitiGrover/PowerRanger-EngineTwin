import time

import numpy as np
import pytest

from enginetwin.latency import (
    Stream,
    alert_latency,
    predict_with_cnn,
    predict_with_features,
    predict_with_obd,
    simulate_onset,
)
from enginetwin.simulator import simulate_window

FS = 16000
OBD = {k: 0.0 for k in ("engine_speed", "calculated_load_value", "engine_coolant_temperature", "maf", "timing_advance", "short_term_fuel_trim_bank1", "long_term_fuel_trim_bank1")}


def test_a_stream_is_the_healthy_signal_before_the_onset_and_the_faulty_signal_after_it():
    s = simulate_onset("misfire", 0.8, onset_s=3.0, duration_s=6.0, rpm=2400, seed=5)
    healthy = simulate_window(rpm=2400, seed=5, duration_s=6.0, snr_db=20.0)
    faulty = simulate_window(rpm=2400, fault="misfire", severity=0.8, seed=5, duration_s=6.0, snr_db=20.0)
    cut = 3 * FS

    assert len(s.sound) == len(s.vibration) == 6 * FS and s.fs == FS and s.onset_s == 3.0
    assert np.array_equal(s.sound[:cut], healthy.sound[:cut]) and np.array_equal(s.sound[cut:], faulty.sound[cut:])
    assert np.array_equal(s.vibration[cut:], faulty.vibration[cut:])


def test_the_window_ending_at_a_time_holds_the_latest_seconds_and_the_obd_readings_of_that_moment():
    s = simulate_onset("misfire", 0.8, onset_s=3.0, duration_s=6.0, rpm=2400, seed=5)
    before, after = s.window(end_s=2.5, window_s=2.0), s.window(end_s=4.5, window_s=2.0)

    assert len(before.sound) == 2 * FS and np.array_equal(before.sound, s.sound[int(0.5 * FS) : int(2.5 * FS)])
    assert (before.fault, before.severity) == ("healthy", 0.0) and (after.fault, after.severity) == ("misfire", 0.8)
    assert before.obd["short_term_fuel_trim_bank1"] != after.obd["short_term_fuel_trim_bank1"]


def silent_then_loud(onset_s=3.0, seconds=8.0):
    n = int(seconds * FS)
    sound = np.zeros(n)
    sound[int(onset_s * FS) :] = np.sin(2 * np.pi * 440 * np.arange(n - int(onset_s * FS)) / FS)  # RMS 0.707
    return Stream(sound=sound, vibration=sound, fs=FS, onset_s=onset_s, obd_before=OBD, obd_after=OBD, fault="knock", severity=1.0)


def loud_last_second(window):
    """Stands in for a model that needs a whole second of the fault before it is sure."""
    return "knock" if np.sqrt(np.mean(window.sound[-FS:] ** 2)) > 0.6 else "healthy"


def test_latency_counts_from_the_onset_to_the_end_of_the_chunk_that_confirms_the_alert():
    # Onset 3.0 s. Windows end every 0.5 s from 2.0 s. The last second is fully faulty only for windows ending at 4.0 s
    # or later (RMS 0.707 > 0.6; at 3.5 s it is half faulty, RMS 0.5), so the first alert is at 4.0 and two in a row
    # confirm at 4.5: 1.5 s after the onset.
    r = alert_latency(loud_last_second, silent_then_loud(), chunk_s=0.5, window_s=2.0, confirm=2)

    assert r["detected_at_s"] == pytest.approx(4.5)
    assert r["latency_s"] == pytest.approx(1.5, abs=0.05) and r["latency_s"] >= 1.5
    assert r["false_alarm_rate"] == 0.0


def test_a_single_blip_is_not_confirmed_when_two_in_a_row_are_required():
    calls = iter(["healthy", "healthy", "healthy", "knock", "healthy", "knock", "healthy", "healthy", "healthy"] + ["healthy"] * 20)
    r = alert_latency(lambda w: next(calls), silent_then_loud(), chunk_s=0.5, window_s=2.0, confirm=2)

    assert r["latency_s"] is None


def test_measured_compute_time_is_added_to_the_latency():
    def slow(window):
        time.sleep(0.1)
        return loud_last_second(window)

    r = alert_latency(slow, silent_then_loud(), chunk_s=0.5, window_s=2.0, confirm=2)

    assert r["compute_s"] >= 0.1 and r["latency_s"] == pytest.approx(1.5 + r["compute_s"], abs=0.001)


def test_alarms_before_the_onset_are_counted_as_false_alarms_and_do_not_shorten_the_latency():
    r = alert_latency(lambda w: "knock", silent_then_loud(), chunk_s=0.5, window_s=2.0, confirm=2)

    assert r["false_alarm_rate"] == 1.0  # windows ending at 2.0, 2.5 and 3.0 s were all flagged
    assert r["detected_at_s"] == pytest.approx(4.0)  # only chunks after the onset (3.5, 4.0) count


class Stub:
    def __init__(self):
        self.seen = None

    def predict(self, *args):
        self.seen = args
        return np.array(["misfire"])


def test_adaptors_feed_each_model_the_input_it_was_trained_on():
    w = simulate_window(rpm=2400, seed=1)
    features, obd, cnn = Stub(), Stub(), Stub()
    cnn.predict_proba = lambda spec, o: (setattr(cnn, "seen", (spec, o)), np.array([[0.9, 0.1, 0.1]]))[1]

    assert predict_with_features(features)(w) == "misfire" and features.seen[0].shape == (1, 44)
    assert predict_with_obd(obd)(w) == "misfire" and obd.seen[0].shape == (1, 8)
    assert predict_with_cnn(cnn)(w) == "misfire"
    assert cnn.seen[0].shape == (1, 2, 64, 64) and cnn.seen[1].shape == (1, 8)
    assert abs(cnn.seen[0].mean()) < 1e-3  # the spectrogram arrives standardised, as in training
