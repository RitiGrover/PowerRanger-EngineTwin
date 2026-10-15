import numpy as np
import pytest

from enginetwin.fault_features import feature_matrix
from enginetwin.labelled import make_labelled_set
from enginetwin.severity import estimate_severity, fit_severity_models


@pytest.fixture(scope="module")
def data():
    make = lambda n, seed: make_labelled_set(n, rpm_range=(800, 3000), snr_db_range=(10, 25), seed=seed, healthy_efficiency=lambda r, l: 0.3)
    train, test = make(60, 31), make(30, 32)
    return feature_matrix(train), train, feature_matrix(test), test


def truth(s, fault):
    return np.array([sev if fault in f.split("+") else 0.0 for f, sev in zip(s.fault, s.severity)])


@pytest.mark.parametrize("fault, min_corr", [("misfire", 0.9), ("knock", 0.75), ("valve_wear", 0.9)])
def test_each_fault_severity_is_estimated_on_unseen_windows(data, fault, min_corr):
    X, train, Xt, test = data
    models = fit_severity_models(X, train.fault, train.severity)
    predicted = np.array([estimate_severity(models, x, fault) for x in Xt])
    want = truth(test, fault)

    assert np.corrcoef(predicted, want)[0, 1] > min_corr
    assert np.mean(np.abs(predicted - want)) < 0.15
    assert predicted.min() >= 0.0 and predicted.max() <= 1.0


def test_a_fault_that_is_not_there_is_estimated_near_zero(data):
    X, train, Xt, test = data
    models = fit_severity_models(X, train.fault, train.severity)
    absent = np.array([estimate_severity(models, x, "misfire") for x in Xt[truth(test, "misfire") == 0]])

    assert absent.mean() < 0.1


def test_an_unknown_fault_name_is_refused(data):
    X, train, _, _ = data
    models = fit_severity_models(X, train.fault, train.severity)

    with pytest.raises(ValueError):
        estimate_severity(models, X[0], "overheating")
