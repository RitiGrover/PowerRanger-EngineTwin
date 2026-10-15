import numpy as np
import pytest

from enginetwin.classifiers import fit_softmax, fit_svm, top_features

FITTERS = [fit_svm, fit_softmax]


def clusters(n=60, seed=0, scale=1.0):
    """Three classes, centred at 0, 6 and 12 on feature 0 (unit spread), with a pure-noise feature 1. Feature 0 is
    multiplied by `scale` to check that models do not depend on raw feature units."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k, name in enumerate(["healthy", "misfire", "knock"]):
        X.append(np.column_stack([rng.normal(6 * k, 1, n) * scale, rng.normal(0, 1, n)]))
        y += [name] * n
    return np.vstack(X), np.array(y)


@pytest.mark.parametrize("fit", FITTERS)
def test_well_separated_classes_are_classified_perfectly_on_unseen_data(fit):
    X, y = clusters(seed=0)
    Xt, yt = clusters(seed=1)
    model = fit(X, y)

    assert np.mean(model.predict(Xt) == yt) >= 0.97  # 6-sigma gaps leave only a tail of overlap


@pytest.mark.parametrize("fit", FITTERS)
def test_probabilities_sum_to_one_and_the_largest_matches_the_prediction(fit):
    X, y = clusters()
    model = fit(X, y)
    p = model.predict_proba(X)

    assert p.shape == (len(X), 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.array_equal(model.classes_[p.argmax(axis=1)], model.predict(X))


@pytest.mark.parametrize("fit", FITTERS)
def test_a_point_far_inside_a_class_gets_high_confidence_for_it(fit):
    model = fit(*clusters())

    p = model.predict_proba(np.array([[6.0, 0.0]]))[0]

    assert model.classes_[p.argmax()] == "misfire" and p.max() > 0.9


@pytest.mark.parametrize("fit", FITTERS)
def test_raw_feature_units_do_not_matter(fit):
    X, y = clusters(scale=1e6)  # feature 0 in the millions
    Xt, yt = clusters(seed=1, scale=1e6)

    assert np.mean(fit(X, y).predict(Xt) == yt) >= 0.97


@pytest.mark.parametrize("fit", FITTERS)
def test_the_same_seed_gives_the_same_model(fit):
    X, y = clusters()

    assert np.array_equal(fit(X, y, seed=3).predict_proba(X), fit(X, y, seed=3).predict_proba(X))


def test_softmax_explains_a_class_by_the_features_that_push_toward_it():
    X, y = clusters()
    model = fit_softmax(X, y)

    ranked = top_features(model, ["strength", "noise"], label="knock", n=2)

    assert ranked[0][0] == "strength" and ranked[0][1] > 0  # higher strength pushes towards knock
    assert abs(ranked[0][1]) > 5 * abs(ranked[1][1])  # the noise feature carries almost no weight
