import numpy as np
import pytest

from enginetwin.fault_features import FEATURE_NAMES, feature_matrix
from enginetwin.labelled import make_labelled_set
from enginetwin.obd_ann import OBD_FEATURE_NAMES, fit_obd_ann, obd_matrix
from enginetwin.schema import FAULTS


def tiny_set(n=2, seed=1):
    return make_labelled_set(n, rpm_range=(800, 3000), snr_db_range=(15, 25), seed=seed, healthy_efficiency=lambda r, l: 0.3)


def test_obd_matrix_holds_the_eight_obd_values_in_the_same_order_as_the_full_feature_vector():
    s = tiny_set()
    X = obd_matrix(s)

    assert X.shape == (len(s), 8)
    assert OBD_FEATURE_NAMES == [n for n in FEATURE_NAMES if n.startswith("obd_")]
    assert np.array_equal(X[:, 0], s.obd["engine_speed"])
    assert np.allclose(X[:, -1], s.obd["short_term_fuel_trim_bank1"] + s.obd["long_term_fuel_trim_bank1"])
    assert np.allclose(X, feature_matrix(s)[:, -8:])


def clusters(n=80, seed=0, scale=1.0):
    """Three classes 6 sigma apart on column 0, plus 7 noise columns (an OBD-sized row of 8)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(3 * n, 8))
    y = np.repeat(["healthy", "misfire", "knock"], n)
    X[:, 0] = np.concatenate([rng.normal(6 * k, 1, n) for k in range(3)]) * scale
    return X, y


def test_the_network_separates_well_separated_classes_on_unseen_data():
    X, y = clusters()
    Xt, yt = clusters(seed=1)

    assert np.mean(fit_obd_ann(X, y).predict(Xt) == yt) >= 0.95


def test_probabilities_sum_to_one_and_the_largest_matches_the_prediction():
    X, y = clusters()
    model = fit_obd_ann(X, y)
    p = model.predict_proba(X)

    assert p.shape == (len(X), 3) and np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(model.classes_[p.argmax(axis=1)], model.predict(X))


def test_raw_feature_units_do_not_matter():
    X, y = clusters(scale=1e6)
    Xt, yt = clusters(seed=1, scale=1e6)

    assert np.mean(fit_obd_ann(X, y).predict(Xt) == yt) >= 0.95


def test_the_same_seed_gives_the_same_network():
    X, y = clusters()

    assert np.array_equal(fit_obd_ann(X, y, seed=3).predict_proba(X), fit_obd_ann(X, y, seed=3).predict_proba(X))


def test_the_network_is_small_enough_for_a_laptop():
    model = fit_obd_ann(*clusters())

    assert model.n_parameters < 10_000


def test_obd_alone_finds_misfire_but_barely_sees_knock_and_not_at_all_valve_wear():
    # Misfire moves the fuel trim by about 8% x severity. Knock only pulls timing back about 6 degrees x severity against
    # a 6.6 degree spread (wide on purpose, as in the real data). Valve wear is mechanical: every OBD reading is untouched.
    # "Finds" is judged per fault family, so misfire+knock counts as a misfire window.
    train, test = tiny_set(120, seed=11), tiny_set(60, seed=12)
    predicted = fit_obd_ann(obd_matrix(train), train.fault, seed=0).predict(obd_matrix(test))

    def lift(family):
        """Share of the family's windows flagged minus share of other windows flagged (0 = no better than guessing)."""
        flagged = np.array([family in p for p in predicted])
        present = np.array([family in f for f in test.fault])
        return flagged[present].mean() - flagged[~present].mean()

    assert lift("misfire") > 0.35
    assert lift("knock") < 0.2
    assert abs(lift("valve_wear")) < 0.15
