import numpy as np
import pytest

from enginetwin.cnn import faults_to_multihot, fit_cnn, labels_from_proba, occlusion_map

PATCHES = [(0, 8, 8), (0, 40, 24), (1, 24, 48)]  # (channel, row, column) of the 8x8 bright patch each fault leaves


def synth(n, seed, obd_signal=False):
    """Noise images; fault k brightens its own 8x8 patch. Any subset of the three faults can be present together."""
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 3)) < 0.5).astype(float)
    spec = rng.normal(size=(n, 2, 64, 64)).astype(np.float32)
    obd = rng.normal(size=(n, 8))
    for k, (c, r, col) in enumerate(PATCHES):
        if obd_signal:
            obd[:, k] += 4 * y[:, k]  # the label is only in the OBD columns
        else:
            spec[y[:, k] == 1, c, r : r + 8, col : col + 8] += 3.0
    return spec, obd, y


def test_labels_become_one_flag_per_fault_so_combined_faults_set_two_flags():
    flags = faults_to_multihot(["healthy", "misfire", "knock", "valve_wear", "misfire+knock", "knock+valve_wear"])

    assert flags.tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]  # misfire, knock, valve_wear


def test_flags_above_the_threshold_turn_back_into_the_seven_labels_and_never_more_than_two_faults():
    proba = np.array([[0.1, 0.2, 0.3], [0.9, 0.1, 0.2], [0.8, 0.7, 0.1], [0.2, 0.6, 0.9], [0.9, 0.8, 0.7]])

    assert list(labels_from_proba(proba)) == ["healthy", "misfire", "misfire+knock", "knock+valve_wear", "misfire+knock"]


def test_the_cnn_finds_each_fault_in_its_own_place_and_two_at_once():
    spec, obd, y = synth(400, seed=0)
    spec_t, obd_t, y_t = synth(200, seed=1)
    model = fit_cnn(spec, obd, y, seed=0, epochs=20)
    flags = model.predict_proba(spec_t, obd_t) > 0.5

    assert np.all((flags == y_t).mean(axis=0) > 0.95)
    both = y_t.sum(axis=1) == 2
    assert both.sum() > 20 and np.all(flags[both] == y_t[both].astype(bool), axis=1).mean() > 0.9


def test_probabilities_are_per_fault_sigmoids_between_0_and_1():
    spec, obd, y = synth(64, seed=0)
    p = fit_cnn(spec, obd, y, epochs=1).predict_proba(spec, obd)

    assert p.shape == (64, 3) and p.min() >= 0 and p.max() <= 1


def test_the_obd_branch_is_used_when_the_label_is_only_in_the_obd_values():
    spec, obd, y = synth(400, seed=0, obd_signal=True)
    spec_t, obd_t, y_t = synth(200, seed=1, obd_signal=True)
    flags = fit_cnn(spec, obd, y, seed=0, epochs=40).predict_proba(spec_t, obd_t) > 0.5

    assert np.all((flags == y_t).mean(axis=0) > 0.9)


def test_the_same_seed_gives_the_same_network_and_it_is_small():
    spec, obd, y = synth(64, seed=0)
    a, b = fit_cnn(spec, obd, y, seed=2, epochs=1), fit_cnn(spec, obd, y, seed=2, epochs=1)

    assert np.array_equal(a.predict_proba(spec, obd), b.predict_proba(spec, obd))
    assert a.n_parameters < 20_000


def test_the_occlusion_map_lights_up_the_patch_that_triggered_the_fault():
    spec, obd, y = synth(400, seed=0)
    model = fit_cnn(spec, obd, y, seed=0, epochs=20)
    on = np.flatnonzero(y[:, 1] == 1)[0]  # a window where fault 1 ("knock") is present

    heat = occlusion_map(model, spec[on], obd[on], fault="knock", patch=8)

    assert heat.shape == (64, 64)
    r, c = np.unravel_index(heat.argmax(), heat.shape)
    assert 40 <= r < 48 and 24 <= c < 32  # inside knock's patch (channel 0, rows 40-47, columns 24-31)
    outside = np.ones((64, 64), bool)
    outside[40:48, 24:32] = False
    assert heat[outside].mean() < 0.2 * heat.max()
