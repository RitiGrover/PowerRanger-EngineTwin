import numpy as np
import pytest

from enginetwin.fault_features import feature_matrix
from enginetwin.health import fit_severity_regressor, health_index, time_to_service
from enginetwin.labelled import make_labelled_set


def test_health_is_100_for_a_new_valve_and_0_for_a_fully_worn_one():
    assert health_index(0.0) == 100.0 and health_index(1.0) == 0.0
    assert health_index(np.array([0.25, 0.5])).tolist() == [75.0, 50.0]


def test_time_to_service_extrapolates_the_recent_trend_to_the_threshold():
    # health = 100 - 2 x day. On day 10 it is 80 and falls 2 a day, so it reaches 20 after (80 - 20) / 2 = 30 more days.
    days = np.arange(11)

    assert time_to_service(days, 100 - 2.0 * days, threshold=20, last=11) == pytest.approx(30.0)


def test_time_to_service_uses_only_the_recent_days_when_the_decline_speeds_up():
    days = np.arange(21)
    health = np.where(days <= 10, 100.0, 100.0 - 5.0 * (days - 10))  # flat, then 5 a day: at day 20 it is 50

    assert time_to_service(days, health, threshold=20, last=6) == pytest.approx(6.0)  # (50 - 20) / 5


def test_a_steady_noisy_decline_gives_an_estimate_close_to_the_true_time():
    rng = np.random.default_rng(0)
    days = np.arange(30)
    health = 100 - 1.5 * days + rng.normal(0, 1.0, 30)  # at day 29: 56.5 true, 36.5 above 20, 24.3 days to go

    assert time_to_service(days, health, threshold=20, last=30) == pytest.approx(24.3, rel=0.15)


def test_no_decline_means_no_service_is_predicted_and_an_engine_already_below_the_threshold_needs_it_now():
    days = np.arange(10)

    assert time_to_service(days, np.full(10, 90.0), threshold=20, last=10) == np.inf
    assert time_to_service(days, 100 - 10.0 * days, threshold=20, last=10) == 0.0  # health is 10 at the end


@pytest.fixture(scope="module")
def wear_data():
    make = lambda seed: make_labelled_set(40, rpm_range=(800, 3000), snr_db_range=(15, 25), seed=seed, healthy_efficiency=lambda r, l: 0.3)
    train, test = make(21), make(22)
    target = lambda s: np.array([sev if "valve_wear" in f else 0.0 for f, sev in zip(s.fault, s.severity)])
    return feature_matrix(train), target(train), feature_matrix(test), target(test)


def test_wear_is_estimated_from_the_valve_features_on_unseen_windows(wear_data):
    X, y, Xt, yt = wear_data
    predicted = fit_severity_regressor(X, y).predict(Xt)

    assert np.mean(np.abs(predicted - yt)) < 0.15
    assert np.corrcoef(predicted, yt)[0, 1] > 0.85
    assert predicted.min() >= 0 and predicted.max() <= 1


def test_a_misfire_or_knock_without_wear_is_not_read_as_valve_wear(wear_data):
    X, y, Xt, yt = wear_data
    predicted = fit_severity_regressor(X, y).predict(Xt)

    assert predicted[yt == 0].mean() < 0.1
