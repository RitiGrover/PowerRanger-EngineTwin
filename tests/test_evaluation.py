import pytest

from enginetwin.evaluation import evaluate

LABELS = ["A", "B", "C"]
TRUE = ["A", "A", "A", "B", "B", "C"]
PRED = ["A", "A", "B", "B", "B", "A"]


def test_accuracy_and_confusion_matrix_match_a_hand_count():
    r = evaluate(TRUE, PRED, LABELS)

    assert r["accuracy"] == pytest.approx(4 / 6)
    assert r["confusion"].tolist() == [[2, 1, 0], [0, 2, 0], [1, 0, 0]]  # rows true, columns predicted


def test_macro_f1_averages_the_per_class_f1_including_a_class_that_is_never_found():
    # A: precision 2/3, recall 2/3 -> 2/3.  B: precision 2/3, recall 1 -> 4/5.  C: never predicted -> 0.
    r = evaluate(TRUE, PRED, LABELS)

    assert r["f1"] == pytest.approx({"A": 2 / 3, "B": 0.8, "C": 0.0})
    assert r["macro_f1"] == pytest.approx((2 / 3 + 0.8 + 0) / 3)


def test_a_perfect_prediction_scores_one():
    r = evaluate(TRUE, TRUE, LABELS)

    assert r["accuracy"] == 1.0 and r["macro_f1"] == 1.0
