import numpy as np
import pytest

from enginetwin.autoencoder import anomaly_auroc, anomaly_threshold, fit_autoencoder


def healthy(n, seed, dims=10, factors=3):
    """Healthy 'engine': ten readings that move together through three hidden factors, plus a little noise."""
    rng = np.random.default_rng(seed)
    mixing = np.random.default_rng(99).normal(size=(factors, dims))
    return rng.normal(size=(n, factors)) @ mixing + 0.05 * rng.normal(size=(n, dims))


def faulty(n, seed, dims=10):
    """Readings that break the healthy relationships (as a new kind of fault would): independent values."""
    return np.random.default_rng(seed).normal(size=(n, dims)) * 2.0


def test_the_autoencoder_scores_readings_that_break_the_healthy_pattern_far_higher():
    model = fit_autoencoder(healthy(2000, 0), seed=0)

    assert anomaly_auroc(model.score(healthy(500, 1)), model.score(faulty(500, 2))) > 0.95


def test_a_threshold_at_the_99th_percentile_of_healthy_scores_flags_about_1_percent_of_new_healthy_windows():
    model = fit_autoencoder(healthy(2000, 0), seed=0)
    threshold = anomaly_threshold(model.score(healthy(2000, 1)), q=0.99)

    assert np.mean(model.score(healthy(4000, 3)) > threshold) < 0.03
    assert np.mean(model.score(faulty(500, 2)) > threshold) > 0.9


def test_the_threshold_is_the_requested_quantile_of_the_scores():
    assert anomaly_threshold(np.arange(101.0), q=0.99) == pytest.approx(99.0)


def test_auroc_counts_the_share_of_fault_healthy_pairs_the_fault_scores_higher_in():
    # healthy 1,2,3 vs fault 2.5,4,5: 2.5 beats two of three, 4 and 5 beat all three: 8 of 9 pairs.
    assert anomaly_auroc([1, 2, 3], [2.5, 4, 5]) == pytest.approx(8 / 9)


def test_scores_are_never_negative_the_same_seed_gives_the_same_model_and_it_is_small():
    X = healthy(300, 0)
    a, b = fit_autoencoder(X, seed=4), fit_autoencoder(X, seed=4)

    assert np.all(a.score(X) >= 0) and np.array_equal(a.score(X), b.score(X))
    assert a.n_parameters < 5000


def test_raw_units_do_not_matter_because_inputs_are_standardised_on_the_healthy_data():
    X = healthy(2000, 0)
    scale = np.array([1e6] + [1.0] * 9)
    model = fit_autoencoder(X * scale, seed=0)

    assert anomaly_auroc(model.score(healthy(500, 1) * scale), model.score(faulty(500, 2) * scale)) > 0.95


def test_the_healthy_threshold_uses_only_the_healthy_rows_of_a_validation_set():
    from enginetwin.autoencoder import healthy_threshold

    model = fit_autoencoder(healthy(2000, 0), seed=0)
    X = np.vstack([healthy(1000, 1), faulty(1000, 2)])
    labels = np.array(["healthy"] * 1000 + ["misfire"] * 1000)

    assert healthy_threshold(model, X, labels, q=0.99) == pytest.approx(anomaly_threshold(model.score(healthy(1000, 1)), 0.99))
