from collections import Counter

import numpy as np
import pytest

from enginetwin.dataset import make_dataset, make_splits
from enginetwin.schema import FAULTS
from enginetwin.signals import bandpass

RPMS = [900, 1800, 2700]  # 900 apart: engine_speed noise (sd 8) and the largest misfire dip (90) cannot blur them


def near(window, rpms):
    return min(rpms, key=lambda r: abs(r - window.obd["engine_speed"]))


def test_the_dataset_has_the_same_number_of_windows_for_every_fault_label():
    windows = make_dataset(n_per_class=3, rpms=RPMS, seed=1, duration_s=0.5)

    assert Counter(w.fault for w in windows) == {fault: 3 for fault in FAULTS}
    assert all(w.source == "simulated" and w.duration_s == 0.5 for w in windows)


def test_every_window_runs_at_one_of_the_requested_speeds_and_each_speed_is_used():
    windows = make_dataset(n_per_class=3, rpms=RPMS, seed=1, duration_s=0.5)

    healthy = [w for w in windows if w.fault == "healthy"]
    assert sorted(near(w, RPMS) for w in healthy) == RPMS  # 3 windows over 3 speeds
    for w in windows:
        assert abs(w.obd["engine_speed"] - near(w, RPMS)) < 150


def test_severity_is_zero_when_healthy_and_clearly_present_otherwise():
    windows = make_dataset(n_per_class=6, rpms=RPMS, seed=2, duration_s=0.5)

    assert all(w.severity == 0.0 for w in windows if w.fault == "healthy")
    assert all(0.3 <= w.severity <= 1.0 for w in windows if w.fault != "healthy")


def test_same_seed_gives_the_same_dataset_and_windows_within_it_differ():
    a = make_dataset(n_per_class=2, rpms=RPMS, seed=3, duration_s=0.5)
    b = make_dataset(n_per_class=2, rpms=RPMS, seed=3, duration_s=0.5)
    c = make_dataset(n_per_class=2, rpms=RPMS, seed=4, duration_s=0.5)

    assert all(np.array_equal(x.sound, y.sound) for x, y in zip(a, b))
    assert not np.array_equal(a[0].sound, c[0].sound)
    healthy = [w for w in a if w.fault == "healthy"]
    assert not np.array_equal(healthy[0].sound, healthy[1].sound)


def test_misfire_windows_do_not_all_hit_the_same_cylinder():
    windows = make_dataset(n_per_class=12, rpms=[1200], seed=5, duration_s=1.0)
    from enginetwin.signals import per_cylinder_energy

    weakest = {
        min((e := per_cylinder_energy(w.vibration, w.vibration_fs, 1200)), key=e.get)
        for w in windows
        if w.fault == "misfire" and w.severity > 0.8
    }
    assert len(weakest) > 1


def test_splits_hold_out_speeds_and_add_a_noisy_test_set():
    train_rpms, test_rpms = [900, 2700], [1800]

    splits = make_splits(train_rpms, test_rpms, n_train=2, n_test=2, seed=6, noisy_sigma=0.05, duration_s=0.5)

    assert {near(w, train_rpms + test_rpms) for w in splits.train} == set(train_rpms)
    assert {near(w, train_rpms + test_rpms) for w in splits.test_unseen_rpm} == set(test_rpms)
    assert {near(w, train_rpms + test_rpms) for w in splits.test_noisy} == set(train_rpms)  # noise only, speeds as trained
    assert len(splits.train) == 2 * len(FAULTS) and len(splits.test_noisy) == 2 * len(FAULTS)


def test_the_noisy_test_set_is_noisier_than_the_training_set():
    splits = make_splits([900, 2700], [1800], n_train=2, n_test=2, seed=6, noisy_sigma=0.05, duration_s=0.5)

    def hiss(windows):  # 7-8 kHz holds almost nothing but noise in a healthy window
        return np.mean([np.mean(bandpass(w.sound, w.sound_fs, 7000, 7900) ** 2) for w in windows if w.fault == "healthy"])

    assert hiss(splits.test_noisy) > 10 * hiss(splits.train)


def test_speeds_must_not_overlap_between_train_and_unseen_test():
    with pytest.raises(ValueError, match="overlap"):
        make_splits([900, 1800], [1800], n_train=1, n_test=1, seed=1, duration_s=0.5)


def test_no_window_is_shared_between_splits():
    splits = make_splits([900, 2700], [1800], n_train=2, n_test=2, seed=6, duration_s=0.5)

    train_sounds = [w.sound for w in splits.train]
    for w in splits.test_noisy + splits.test_unseen_rpm:
        assert not any(np.array_equal(w.sound, s) for s in train_sounds)
