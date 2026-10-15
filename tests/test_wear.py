import numpy as np

from enginetwin.signals import band_energy_fraction
from enginetwin.wear import wear_series


def valve_share(w):
    return band_energy_fraction(w.vibration, w.vibration_fs, 2500, 3500)


def test_a_wear_series_has_one_window_per_day_at_the_same_operating_point():
    series = wear_series(n_days=30, seed=1, rpm=2200, load=0.45)

    assert list(series.days) == list(range(30))
    assert len(series.windows) == 30
    assert all(w.source == "simulated" for w in series.windows)
    rpms = [round(w.obd["engine_speed"] / 100) for w in series.windows]  # OBD noise is a few rpm
    assert set(rpms) == {22}


def test_wear_starts_at_a_healthy_engine_and_ends_near_fully_worn():
    series = wear_series(n_days=60, seed=1)
    first, last = series.windows[0], series.windows[-1]

    assert (first.fault, first.severity) == ("healthy", 0.0)
    assert last.fault == "valve_wear"
    assert 0.9 <= last.severity <= 1.0


def test_wear_only_ever_grows_and_grows_slowly():
    # A 60-day power-law ramp with exponent 1.5 climbs at most 1.5/59 = 0.025 a day; allow jitter up to 0.06.
    severity = np.array([w.severity for w in wear_series(n_days=60, seed=3).windows])

    assert np.all(np.diff(severity) >= 0)
    assert np.diff(severity).max() < 0.06


def test_the_valve_tick_energy_in_the_vibration_grows_as_the_valve_wears():
    series = wear_series(n_days=40, seed=2)
    early = np.mean([valve_share(w) for w in series.windows[:5]])
    late = np.mean([valve_share(w) for w in series.windows[-5:]])

    assert late > 5 * early


def test_the_same_seed_gives_the_same_series_and_a_different_seed_a_different_one():
    a = wear_series(n_days=10, seed=7)
    b = wear_series(n_days=10, seed=7)
    c = wear_series(n_days=10, seed=8)

    assert np.array_equal(a.windows[5].vibration, b.windows[5].vibration)
    assert [w.severity for w in a.windows] == [w.severity for w in b.windows]
    assert [w.severity for w in a.windows] != [w.severity for w in c.windows]
