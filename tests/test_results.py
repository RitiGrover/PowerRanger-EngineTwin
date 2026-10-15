import json

import numpy as np
import pytest

from enginetwin.results import build_results_table, latency_table, load_confusions


@pytest.fixture
def cache(tmp_path):
    cell = lambda a, f: {"accuracy": a, "macro_f1": f, "confusion": [[1, 0], [0, 1]]}
    (tmp_path / "results_classical.json").write_text(
        json.dumps(
            {
                "svm": {"test_same": cell(0.99, 0.98), "test_heavy_noise": cell(0.14, 0.04)},
                "softmax": {"test_same": cell(0.97, 0.96)},
                "softmax_top_features": {"misfire": [["x", 1.0]]},
            }
        )
    )
    (tmp_path / "results_obd_ann.json").write_text(
        json.dumps({"test_same": cell(0.27, 0.26), "misfire_family_detection": {"found": 0.76, "false_alarm": 0.14}})
    )
    (tmp_path / "results_cnn.json").write_text(json.dumps({"test_same": {**cell(0.90, 0.89), "per_fault_recall": {}}}))
    (tmp_path / "results_latency.json").write_text(
        json.dumps({"svm": {"misfire": {"mean_latency_s": 1.4, "detected": 0.9}}, "cnn": {"misfire": {"mean_latency_s": 1.1, "detected": 1.0}}})
    )
    return tmp_path


def test_every_model_and_split_becomes_one_row_with_accuracy_and_macro_f1(cache):
    table = build_results_table(cache)
    row = lambda m, s: table[(table.model == m) & (table.split == s)].iloc[0]

    assert len(table) == 5  # svm x2, softmax, obd_ann, cnn; extras such as family detection are not rows
    assert row("svm", "test_heavy_noise").accuracy == 0.14 and row("obd_ann", "test_same").macro_f1 == 0.26
    assert row("cnn", "test_same").accuracy == 0.90 and list(table.columns) == ["model", "split", "accuracy", "macro_f1"]


def test_confusion_matrices_are_kept_per_model_and_split(cache):
    confusions = load_confusions(cache)

    assert np.array_equal(confusions[("svm", "test_same")], [[1, 0], [0, 1]]) and ("cnn", "test_same") in confusions


def test_the_latency_table_has_one_row_per_model_and_fault(cache):
    t = latency_table(cache)

    assert sorted(zip(t.model, t.fault)) == [("cnn", "misfire"), ("svm", "misfire")]
    assert t[t.model == "cnn"].mean_latency_s.iloc[0] == 1.1


def test_a_model_whose_results_file_is_missing_is_simply_left_out(cache):
    (cache / "results_cnn.json").unlink()

    assert "cnn" not in set(build_results_table(cache).model)


def test_the_autoencoder_table_gives_the_false_alarm_rate_and_auroc_per_fault_at_each_split(tmp_path):
    from enginetwin.results import autoencoder_table

    (tmp_path / "results_health_autoencoder.json").write_text(
        json.dumps(
            {
                "autoencoder": {
                    "threshold": 1.0,
                    "test_same": {"false_alarm": 0.02, "per_fault": {"misfire": {"auroc": 0.99, "flagged": 0.95}}},
                    "test_noise_0db": {"false_alarm": 0.30, "per_fault": {"misfire": {"auroc": 0.80, "flagged": 0.6}}},
                },
                "wear_regressor_mae_test_same": 0.02,
            }
        )
    )
    t = autoencoder_table(tmp_path)

    assert sorted(t.split) == ["test_noise_0db", "test_same"]
    row = t[t.split == "test_noise_0db"].iloc[0]
    assert (row.fault, row.auroc, row.flagged, row.false_alarm) == ("misfire", 0.80, 0.6, 0.30)
