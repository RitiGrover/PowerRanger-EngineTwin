import numpy as np
import pandas as pd
import pytest

from enginetwin.losses import efficiency_loss
from enginetwin.labelled import (
    cached_labelled_splits,
    healthy_efficiency_surface,
    load_labelled_set,
    make_labelled_set,
    make_labelled_splits,
    save_labelled_set,
)
from enginetwin.schema import FAULTS
from enginetwin.signals import band_energy_fraction

FLAT = lambda rpm, load: 0.30  # a healthy engine that is 30% efficient everywhere
SMALL = dict(rpm_range=(800, 3000), snr_db_range=(15, 25), seed=1, healthy_efficiency=FLAT)


def small_set(n_per_class=3, **overrides):
    return make_labelled_set(n_per_class, **{**SMALL, **overrides})


def test_the_set_holds_the_same_number_of_windows_for_every_fault_label():
    s = small_set()

    assert len(s) == 3 * len(FAULTS)
    for fault in FAULTS:
        assert np.sum(s.fault == fault) == 3
    assert s.sound.shape == s.vibration.shape == (len(s), 32000)
    assert s.sound.dtype == np.float16
    assert s.fs == 16000


def test_a_record_turns_back_into_a_simulated_window_with_its_label_and_obd_values():
    s = small_set()
    w = s.window(4)

    assert (w.fault, w.source) == (s.fault[4], "simulated")
    assert w.severity == pytest.approx(s.severity[4])
    assert len(w.sound) == 32000 and w.sound_fs == 16000
    assert w.obd["engine_speed"] == pytest.approx(s.obd["engine_speed"][4])
    assert set(s.obd) >= {"engine_speed", "maf", "timing_advance", "short_term_fuel_trim_bank1"}


def test_every_class_is_run_at_exactly_the_same_speeds_and_loads_so_operating_point_cannot_give_the_label_away():
    s = small_set(n_per_class=5)
    by_class = {f: (s.rpm[s.fault == f], s.load[s.fault == f], s.snr_db[s.fault == f]) for f in FAULTS}

    reference = by_class["healthy"]
    for fault in FAULTS:
        for got, want in zip(by_class[fault], reference):
            assert np.array_equal(got, want)


def test_speeds_loads_and_noise_levels_stay_inside_the_requested_ranges_and_vary():
    s = small_set(n_per_class=20)

    assert 800 <= s.rpm.min() and s.rpm.max() < 3000 and s.rpm.std() > 300
    assert 0.2 <= s.load.min() and s.load.max() <= 0.8 and s.load.std() > 0.05
    assert 15 <= s.snr_db.min() and s.snr_db.max() <= 25


def test_severity_is_zero_when_healthy_and_clearly_present_otherwise():
    s = small_set(n_per_class=10)

    assert np.all(s.severity[s.fault == "healthy"] == 0.0)
    assert np.all((s.severity[s.fault != "healthy"] >= 0.3) & (s.severity[s.fault != "healthy"] <= 1.0))


def test_efficiency_truth_is_the_healthy_efficiency_less_the_assumed_fault_loss():
    s = small_set()

    for i in range(len(s)):
        assert s.efficiency[i] == pytest.approx(0.30 * (1 - efficiency_loss(s.fault[i], s.severity[i])), rel=1e-6)


def test_loss_rates_given_to_the_generator_change_the_efficiency_truth():
    s = small_set(loss_rates={"misfire": 0.5})
    i = int(np.flatnonzero(s.fault == "misfire")[0])

    assert s.efficiency[i] == pytest.approx(0.30 * (1 - 0.5 * s.severity[i]), rel=1e-6)


def test_valve_wear_windows_vary_in_which_valves_are_worn():
    s = small_set(n_per_class=30)
    counts = {int(np.count_nonzero(w)) for w in s.worn_valves[s.fault == "valve_wear"]}  # padded with 0

    assert counts == {1, 2}


def test_the_same_seed_gives_the_same_set_and_a_different_seed_a_different_one():
    a, b, c = small_set(), small_set(), small_set(seed=2)

    assert np.array_equal(a.sound, b.sound) and np.array_equal(a.severity, b.severity)
    assert not np.array_equal(a.sound, c.sound)


def test_a_saved_set_loads_back_identical(tmp_path):
    s = small_set()
    save_labelled_set(s, tmp_path / "s.npz")
    back = load_labelled_set(tmp_path / "s.npz")

    assert np.array_equal(back.sound, s.sound) and np.array_equal(back.vibration, s.vibration)
    assert np.array_equal(back.fault, s.fault) and np.array_equal(back.severity, s.severity)
    assert np.array_equal(back.obd["maf"], s.obd["maf"]) and np.array_equal(back.efficiency, s.efficiency)
    assert np.array_equal(back.worn_valves, s.worn_valves) and back.fs == s.fs


SPLIT_ARGS = dict(n_train=3, n_val=2, n_test=2, seed=5, healthy_efficiency=FLAT)


SPLITS = (
    "train", "validation", "validation_wide", "test_same", "test_unseen_rpm", "test_noise_15db", "test_heavy_noise", "test_noise_0db",
)


def test_there_are_eight_splits_and_the_unseen_speed_split_is_held_out_of_all_the_training_conditions():
    sp = make_labelled_splits(**SPLIT_ARGS)

    assert sp._fields == SPLITS
    assert [len(x) for x in sp] == [3 * 7] + [2 * 7] * 7
    for seen in (sp.train, sp.validation, sp.test_same, sp.test_noise_15db, sp.test_heavy_noise, sp.test_noise_0db):
        assert seen.rpm.max() < 3000
    assert sp.test_unseen_rpm.rpm.min() >= 3000 and sp.test_unseen_rpm.rpm.max() <= 5000


def test_the_wide_validation_split_covers_trained_and_unseen_speeds_for_calibration():
    sp = make_labelled_splits(n_train=2, n_val=12, n_test=2, seed=5, healthy_efficiency=FLAT)

    assert sp.validation_wide.rpm.min() >= 800 and sp.validation_wide.rpm.max() <= 5000
    assert sp.validation_wide.rpm.max() > 3000 and sp.validation_wide.rpm.min() < 3000


def test_training_noise_is_drawn_from_10_to_25_db_and_each_test_noise_level_is_fixed_outside_that_draw():
    sp = make_labelled_splits(**SPLIT_ARGS)

    for trained in (sp.train, sp.validation, sp.test_same, sp.test_unseen_rpm):
        assert 10 <= trained.snr_db.min() and trained.snr_db.max() <= 25
    assert 5 <= sp.validation_wide.snr_db.min() and sp.validation_wide.snr_db.max() <= 25
    assert np.all(sp.test_noise_15db.snr_db == 15.0)
    assert np.all(sp.test_heavy_noise.snr_db == 5.0)
    assert np.all(sp.test_noise_0db.snr_db == 0.0)


def test_the_training_noise_level_varies_from_window_to_window():
    s = make_labelled_splits(n_train=20, n_val=1, n_test=1, seed=5, healthy_efficiency=FLAT).train

    assert s.snr_db.std() > 3.0  # a uniform spread over 15 dB has a standard deviation of 4.3


def test_lower_snr_windows_really_are_noisier():
    sp = make_labelled_splits(n_train=3, n_val=2, n_test=4, seed=5, healthy_efficiency=FLAT)
    top_band = lambda s: np.mean(
        [band_energy_fraction(s.window(i).sound, 16000, 7500, 8000) for i in np.flatnonzero(s.fault == "healthy")]
    )

    assert top_band(sp.test_heavy_noise) > 5 * top_band(sp.test_same)
    assert top_band(sp.test_noise_0db) > 2 * top_band(sp.test_heavy_noise)  # 5 dB lower is a 3.2x power step


def test_no_window_is_shared_between_splits():
    sp = make_labelled_splits(**SPLIT_ARGS)
    fingerprints = [{s.sound[i].tobytes() for i in range(len(s))} for s in sp]

    assert sum(len(f) for f in fingerprints) == len(set().union(*fingerprints))


def test_the_cache_builds_once_and_reuses_the_file_for_identical_parameters(tmp_path):
    first = cached_labelled_splits(tmp_path, **SPLIT_ARGS)
    files = list(tmp_path.glob("*.npz"))
    stamp = files[0].stat().st_mtime_ns
    again = cached_labelled_splits(tmp_path, **SPLIT_ARGS)

    assert len(files) == 1 and files[0].stat().st_mtime_ns == stamp
    assert np.array_equal(first.test_same.sound, again.test_same.sound)
    assert np.array_equal(first.train.efficiency, again.train.efficiency)


def test_changing_any_parameter_gives_a_separate_cache_entry(tmp_path):
    cached_labelled_splits(tmp_path, **SPLIT_ARGS)
    cached_labelled_splits(tmp_path, **{**SPLIT_ARGS, "seed": 6})
    cached_labelled_splits(tmp_path, **SPLIT_ARGS, loss_rates={"misfire": 0.3})

    assert len(list(tmp_path.glob("*.npz"))) == 3


def test_the_training_snr_range_is_part_of_the_cache_key(tmp_path, monkeypatch):
    import enginetwin.labelled as labelled

    cached_labelled_splits(tmp_path, **SPLIT_ARGS)
    monkeypatch.setattr(labelled, "NORMAL_SNR_DB", (12.0, 25.0))
    cached_labelled_splits(tmp_path, **SPLIT_ARGS)

    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_the_healthy_surface_reads_efficiency_from_the_nearest_map_cell():
    cells = pd.DataFrame(
        {"rpm": [1000.0, 1000.0, 3000.0, 3000.0], "torque": [20.0, 100.0, 20.0, 100.0], "efficiency": [0.10, 0.20, 0.25, 0.35]}
    )
    surface = healthy_efficiency_surface(cells, max_torque_nm=100.0)

    assert surface(1000, 0.2) == pytest.approx(0.10)  # torque = 0.2 x 100 = 20 Nm
    assert surface(3100, 0.95) == pytest.approx(0.35)
    assert surface(1200, 0.9) == pytest.approx(0.20)
