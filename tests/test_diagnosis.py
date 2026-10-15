import numpy as np
import pytest

from enginetwin.autoencoder import fit_autoencoder
from enginetwin.classifiers import fit_softmax
from enginetwin.cnn import faults_to_multihot, fit_cnn
from enginetwin.diagnosis import Diagnoser
from enginetwin.explain import healthy_baseline
from enginetwin.fault_features import feature_matrix
from enginetwin.labelled import make_labelled_set
from enginetwin.noise import estimate_noise_db, fit_noise_thresholds
from enginetwin.obd_ann import obd_matrix
from enginetwin.severity import fit_severity_models
from enginetwin.simulator import simulate_window
from enginetwin.spectrograms import spectrogram_stack


@pytest.fixture(scope="module")
def diagnoser():
    """A small but real set of models, trained on 50 simulated windows per fault label."""
    s = make_labelled_set(50, rpm_range=(800, 3000), snr_db_range=(10, 25), seed=41, healthy_efficiency=lambda r, l: 0.3)
    X = feature_matrix(s)
    healthy = s.fault == "healthy"
    ae = fit_autoencoder(X[healthy], seed=0)
    noise = np.array([estimate_noise_db(s.window(i)) for i in range(len(s))])
    thresholds = fit_noise_thresholds(ae.score(X), noise, s.fault, edges=(-22.5, -17.5, -12.5, -7.5), q=0.99, min_windows=5)
    return Diagnoser(
        cnn=fit_cnn(spectrogram_stack(s), obd_matrix(s), faults_to_multihot(s.fault), seed=0, epochs=20),
        softmax=fit_softmax(X, s.fault),
        autoencoder=ae,
        thresholds=thresholds,
        severity_models=fit_severity_models(X, s.fault, s.severity),
        baseline=healthy_baseline(X[healthy]),
    )


def test_a_clear_misfire_is_diagnosed_as_a_fault_with_severity_reasons_and_probabilities(diagnoser):
    r = diagnoser.diagnose(simulate_window(rpm=2000, fault="misfire", severity=1.0, seed=901, snr_db=20.0, cylinder=3))

    assert "misfire" in r.label and r.status == "Fault"
    assert 0.5 <= r.confidence <= 1.0
    assert set(r.probabilities) == {"misfire", "knock", "valve_wear"} and all(0 <= p <= 1 for p in r.probabilities.values())
    assert r.probabilities["misfire"] > 0.5 and 0.4 <= r.severities["misfire"] <= 1.0
    assert r.reasons and all(isinstance(s, str) and s for s in r.reasons)
    assert r.anomaly is True and r.obd_available is True and np.isfinite(r.noise_db)


def test_a_healthy_window_is_reported_healthy_with_no_severities(diagnoser):
    r = diagnoser.diagnose(simulate_window(rpm=2000, seed=902, snr_db=20.0))

    assert r.label == "healthy" and r.status == "Healthy" and r.severities == {}
    assert r.reasons == ["all readings are within their normal range"]
    assert max(r.probabilities.values()) < 0.5 and r.confidence > 0.5


def test_two_faults_at_once_are_both_named(diagnoser):
    r = diagnoser.diagnose(simulate_window(rpm=2000, fault="misfire+knock", severity=1.0, seed=903, snr_db=20.0))

    assert r.label == "misfire+knock" and set(r.severities) == {"misfire", "knock"}


def test_the_softmax_seven_label_opinion_is_given_alongside_the_cnn(diagnoser):
    r = diagnoser.diagnose(simulate_window(rpm=2000, fault="knock", severity=1.0, seed=904, snr_db=20.0))

    assert r.softmax_label == "knock" and abs(sum(r.softmax_probabilities.values()) - 1.0) < 1e-6


def test_a_window_without_obd_is_still_diagnosed_and_says_so(diagnoser):
    w = simulate_window(rpm=2000, fault="misfire", severity=1.0, seed=905, snr_db=20.0)
    w.obd["_obd_missing"] = 1.0
    r = diagnoser.diagnose(w)

    assert r.obd_available is False and r.label != ""


def test_the_heat_map_covers_the_spectrogram_and_is_never_negative(diagnoser):
    heat = diagnoser.heatmap(simulate_window(rpm=2000, fault="knock", severity=1.0, seed=906, snr_db=20.0), "knock")

    assert heat.shape == (64, 64) and heat.min() >= 0 and heat.max() > 0


def test_saved_models_load_back_and_diagnose_the_same(diagnoser, tmp_path):
    diagnoser.save(tmp_path)
    w = simulate_window(rpm=2000, fault="misfire", severity=1.0, seed=907, snr_db=20.0)

    again = Diagnoser.load(tmp_path).diagnose(w)
    assert again.label == diagnoser.diagnose(w).label and again.probabilities == diagnoser.diagnose(w).probabilities
