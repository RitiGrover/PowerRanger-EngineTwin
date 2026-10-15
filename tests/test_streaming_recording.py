import io

import numpy as np
import pytest
import soundfile as sf

from enginetwin.latency import simulate_onset
from enginetwin.recording import estimate_rpm_from_audio, has_obd, obd_from_csv, window_from_wav
from enginetwin.simulator import simulate_window
from enginetwin.streaming import rolling_windows

FS = 16000


def wav_bytes(signal, fs, channels=1):
    buffer = io.BytesIO()
    data = np.column_stack([signal] * channels) if channels > 1 else signal
    sf.write(buffer, data, fs, format="WAV")
    return buffer.getvalue()


def test_a_stream_is_cut_into_half_second_steps_each_with_the_latest_two_seconds_and_the_new_chunk():
    s = simulate_onset("knock", 0.8, onset_s=3.0, duration_s=5.0, rpm=2400, seed=1)
    steps = list(rolling_windows(s, chunk_s=0.5, window_s=2.0))

    assert [round(t, 3) for t, *_ in steps] == [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    t, window, sound_chunk, vibration_chunk = steps[3]
    assert np.array_equal(window.sound, s.sound[int(1.5 * FS) : int(3.5 * FS)])
    assert np.array_equal(sound_chunk, s.sound[int(3.0 * FS) : int(3.5 * FS)]) and len(vibration_chunk) == FS // 2
    assert window.fault == "knock" and steps[0][1].fault == "healthy"


def test_a_wav_becomes_a_16_khz_mono_two_second_window_tagged_real():
    w = window_from_wav(wav_bytes(simulate_window(rpm=2400, seed=1).sound * 0.5, 16000))

    assert (w.sound_fs, w.source, len(w.sound), len(w.vibration)) == (16000, "real", 32000, 32000)
    assert np.array_equal(w.sound, w.vibration)  # one microphone: the same signal stands in for both channels


def test_other_sample_rates_are_resampled_and_stereo_is_averaged():
    tone = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(int(2.5 * 44100)) / 44100)
    w = window_from_wav(wav_bytes(tone, 44100, channels=2))
    spectrum = np.abs(np.fft.rfft(w.sound))
    peak_hz = np.fft.rfftfreq(len(w.sound), 1 / w.sound_fs)[spectrum.argmax()]

    assert len(w.sound) == 32000 and peak_hz == pytest.approx(1000, abs=1)


def test_a_longer_recording_is_cut_to_its_loudest_two_seconds():
    recording = 0.01 * np.random.default_rng(0).standard_normal(6 * FS)
    recording[int(3.1 * FS) : int(5.1 * FS)] += 0.5 * np.sin(2 * np.pi * 300 * np.arange(2 * FS) / FS)  # tone from 3.1 to 5.1 s

    w = window_from_wav(wav_bytes(recording, FS))

    assert np.sqrt(np.mean(w.sound**2)) > 0.3  # a window that missed the tone would read about 0.01; half of it 0.25


def test_a_short_recording_is_padded_with_silence_and_a_tiny_one_is_refused():
    short = 0.3 * np.sin(2 * np.pi * 300 * np.arange(FS) / FS)  # 1 s
    w = window_from_wav(wav_bytes(short, FS))

    assert len(w.sound) == 32000 and np.all(w.sound[FS:] == 0)
    with pytest.raises(ValueError):
        window_from_wav(wav_bytes(short[:2000], FS))  # 0.125 s


@pytest.mark.parametrize("rpm", [900.0, 2400.0, 4200.0])
def test_engine_speed_is_found_from_the_audio_alone_for_a_healthy_engine(rpm):
    w = simulate_window(rpm=rpm, seed=2)

    assert estimate_rpm_from_audio(w.sound, w.sound_fs) == pytest.approx(rpm, rel=0.01)


def test_engine_speed_from_audio_is_right_for_nearly_all_windows_of_every_fault_type():
    # Octave errors are the known weakness (a strong misfire can read at half speed), so the claim is a share, not all.
    rng = np.random.default_rng(0)
    right = 0
    for i in range(40):
        rpm = float(rng.uniform(900, 4800))
        fault = ("healthy", "misfire", "knock", "valve_wear")[i % 4]
        w = simulate_window(rpm=rpm, fault=fault, severity=0.0 if fault == "healthy" else float(rng.uniform(0.3, 1.0)), seed=i, snr_db=float(rng.uniform(10, 25)))
        right += abs(estimate_rpm_from_audio(w.sound, w.sound_fs) / rpm - 1) < 0.01

    assert right >= 36


def test_without_obd_the_window_gets_the_audio_speed_neutral_readings_and_a_missing_flag():
    w = window_from_wav(wav_bytes(simulate_window(rpm=2400, seed=1).sound, FS))

    assert w.obd["engine_speed"] == pytest.approx(2400, rel=0.01) and not has_obd(w)
    assert w.obd["short_term_fuel_trim_bank1"] == 0.0 and w.obd["timing_advance"] > 0


def test_an_obd_csv_gives_the_median_of_each_reading_and_marks_obd_present(tmp_path):
    csv = tmp_path / "obd.csv"
    csv.write_text("engine_speed,short_term_fuel_trim_bank1,long_term_fuel_trim_bank1\n2000,5,2\n2200,6,2\n2400,7,2\n9999,50,2\n")
    obd = obd_from_csv(csv)  # medians: 2300, 6.5, 2

    assert obd["engine_speed"] == 2300 and obd["short_term_fuel_trim_bank1"] == 6.5 and obd["long_term_fuel_trim_bank1"] == 2.0
    w = window_from_wav(wav_bytes(simulate_window(rpm=2400, seed=1).sound, FS), obd=obd)
    assert has_obd(w) and w.obd["engine_speed"] == 2300
