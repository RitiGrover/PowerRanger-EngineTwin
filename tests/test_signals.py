import numpy as np
import pytest

from enginetwin.signals import (
    band_energy_fraction,
    bandpass,
    envelope_spectrum,
    half_order_ratio,
    order_amplitude,
    order_spectrum,
    per_cylinder_energy,
    spectrogram,
)

FS = 1000


def sine(freq, amplitude=1.0, seconds=1.0, fs=FS):
    t = np.arange(int(seconds * fs)) / fs
    return amplitude * np.sin(2 * np.pi * freq * t)


# A Hann-windowed sine of amplitude A has its spectral peak at A/2 x mean(window) = A/4 (mean of a Hann window is 1/2).
@pytest.mark.parametrize("rpm, order, freq", [(3000, 1.0, 50), (1200, 0.5, 10), (6000, 2.0, 200)])
def test_order_amplitude_reads_a_sine_at_its_order_whatever_the_rpm(rpm, order, freq):
    assert order_amplitude(sine(freq, amplitude=2.0), FS, rpm, order) == pytest.approx(0.5, rel=0.02)


def test_order_amplitude_is_near_zero_at_an_order_with_no_content():
    assert order_amplitude(sine(50, amplitude=2.0), FS, 3000, 2.0) < 0.01


# order_spectrum: the whole spectrum on an order axis (frequency / crank frequency). Lines line up at any rpm.
@pytest.mark.parametrize("rpm, order", [(1500, 0.5), (1500, 2.0), (3000, 2.0), (1200, 4.0)])
def test_order_spectrum_puts_a_sine_at_its_order_with_the_same_amplitude_at_any_rpm(rpm, order):
    crank_hz = rpm / 60
    signal = sine(order * crank_hz, amplitude=2.0, seconds=2.0)

    orders, amplitudes = order_spectrum(signal, FS, rpm)

    assert orders[int(np.argmax(amplitudes))] == pytest.approx(order)
    assert amplitudes.max() == pytest.approx(0.5, rel=0.02)  # A/4 for a Hann-windowed sine of amplitude 2


def test_order_spectrum_starts_at_order_zero_and_stops_at_max_order():
    orders, amplitudes = order_spectrum(sine(50, seconds=2.0), FS, 3000, max_order=6)

    assert orders[0] == 0.0
    assert orders[-1] == pytest.approx(6.0, abs=0.02)  # 6 x 50 Hz = 300 Hz, 0.5 Hz bins -> 0.01 order steps
    assert len(orders) == len(amplitudes)


# half_order_ratio: misfire indicator, the order-0.5 line relative to the firing line (order 2).
def test_half_order_ratio_is_the_ratio_of_the_two_lines():
    rpm = 1500  # crank 25 Hz: order 0.5 = 12.5 Hz, order 2 = 50 Hz
    signal = sine(12.5, amplitude=1.0, seconds=2.0) + sine(50, amplitude=4.0, seconds=2.0)

    assert half_order_ratio(signal, FS, rpm) == pytest.approx(1 / 4, rel=0.02)


# bandpass and band_energy_fraction: the knock band.
def test_bandpass_keeps_the_sine_inside_the_band_and_removes_the_one_outside():
    mixed = sine(100) + sine(300)

    assert bandpass(mixed, FS, 250, 350) == pytest.approx(sine(300), abs=1e-9)


def test_band_energy_fraction_is_the_share_of_energy_in_the_band():
    # energy of a sine is amplitude^2 / 2 per sample. Amplitudes 1 and 3 give 1 : 9
    mixed = sine(100, amplitude=1.0) + sine(300, amplitude=3.0)

    assert band_energy_fraction(mixed, FS, 250, 350) == pytest.approx(0.9)
    assert band_energy_fraction(mixed, FS, 50, 150) == pytest.approx(0.1)


def test_band_energy_fraction_of_silence_is_zero_not_nan():
    assert band_energy_fraction(np.zeros(FS), FS, 250, 350) == 0.0


# spectrogram: magnitude per (frequency, frame), same A/4 scale as the other spectra.
def test_spectrogram_shows_which_frequency_is_present_in_which_frame():
    signal = np.concatenate([sine(100, seconds=1.0), sine(300, seconds=1.0)])  # 100 Hz then 300 Hz, 2 s at 1 kHz

    # 0.25 s windows -> 4 Hz bins; (2000 - 250) // 250 + 1 = 8 frames, centred 0.125 s, 0.375 s, ...
    freqs, times, magnitude = spectrogram(signal, FS, win_s=0.25, hop_s=0.25)

    assert freqs[1] == pytest.approx(4.0)
    assert len(freqs) == 126
    assert times == pytest.approx(np.arange(8) * 0.25 + 0.125)
    assert magnitude.shape == (126, 8)
    peaks = freqs[magnitude.argmax(axis=0)]
    assert list(peaks) == [100.0] * 4 + [300.0] * 4
    assert magnitude.max() == pytest.approx(0.25, rel=0.02)  # amplitude 1 -> A/4


# envelope_spectrum: repetition rate of resonance ringing (valve ticks), not the resonance itself.
def test_envelope_spectrum_finds_the_modulation_rate_of_a_carrier():
    t = np.arange(FS) / FS
    signal = (1 + 0.5 * np.sin(2 * np.pi * 20 * t)) * np.sin(2 * np.pi * 300 * t)  # 300 Hz ringing, 20 Hz repeat

    freqs, amplitudes = envelope_spectrum(signal, FS)

    assert freqs[int(np.argmax(amplitudes))] == pytest.approx(20.0)
    assert amplitudes.max() == pytest.approx(0.5 / 4, rel=0.02)  # envelope ripple of 0.5 -> A/4 after the Hann window


def test_envelope_spectrum_band_ignores_a_loud_sine_outside_the_band():
    t = np.arange(FS) / FS
    ringing = (1 + 0.5 * np.sin(2 * np.pi * 20 * t)) * np.sin(2 * np.pi * 300 * t)
    signal = ringing + 3 * np.sin(2 * np.pi * 100 * t)

    freqs, amplitudes = envelope_spectrum(signal, FS, band=(250, 350))

    assert freqs[int(np.argmax(amplitudes))] == pytest.approx(20.0)
    assert amplitudes.max() == pytest.approx(0.5 / 4, rel=0.02)


# per_cylinder_energy: firing order 1-3-4-2, one firing every 180 crank degrees = 30/rpm seconds.
def test_per_cylinder_energy_follows_the_firing_order():
    rpm = 1200  # one firing per 0.025 s = 25 samples at 1 kHz; 1 s holds 40 firings, 10 per cylinder
    signal = np.zeros(FS)
    for k, cylinder in enumerate((1, 3, 4, 2) * 10):
        signal[k * 25 + 5] = cylinder  # a spike of height c in cylinder c's own window

    energy = per_cylinder_energy(signal, FS, rpm)

    assert energy == {1: 10 * 1**2, 2: 10 * 2**2, 3: 10 * 3**2, 4: 10 * 4**2}
