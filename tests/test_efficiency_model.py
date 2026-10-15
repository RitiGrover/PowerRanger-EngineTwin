import numpy as np
import pandas as pd
import pytest

from enginetwin.efficiency_model import (
    FEATURES,
    build_xy,
    evaluate_regression,
    feature_importance,
    fit_ann_efficiency,
    fit_gbm_efficiency,
    group_split,
    ocslab_frame,
    ved_frame,
)


def test_the_common_features_are_computed_from_columns_both_datasets_have():
    df = pd.DataFrame(
        {
            "engine_speed": [2000.0, 2000.0], "calculated_load_value": [40.0, 40.0], "vehicle_speed": [50.0, 0.0],
            "short_term_fuel_trim_bank1": [1.5, 0.0], "long_term_fuel_trim_bank1": [2.5, 0.0],
        }
    )
    f = ocslab_frame(df)

    assert list(f[list(FEATURES)].columns) == list(FEATURES)
    assert f["load"].tolist() == [40.0, 40.0]
    assert f["total_fuel_trim"].tolist() == [4.0, 0.0]
    assert f["rpm_per_kmh"].tolist() == [40.0, 400.0]  # 2000 / 50; at standstill the speed is floored at 5 km/h


def test_ved_efficiency_uses_torque_estimated_as_absolute_load_times_the_maximum_torque():
    # 50% load of a 150 Nm engine = 75 Nm at 2000 rpm: 75 x 2000 x 2 pi / 60 = 15708 W. Fuel chosen to burn 52360 W
    # (0.745 g/mL, 43.4 MJ/kg): 52360 / 43.4e6 kg/s = 1.2065e-3 kg/s = 1619.4 uL/s. Efficiency = 15708 / 52360 = 0.30.
    df = pd.DataFrame(
        {
            "engine_speed": [2000.0], "absolute_load": [50.0], "vehicle_speed": [50.0], "fuel_consumption": [1619.4],
            "short_term_fuel_trim_bank1": [0.0], "long_term_fuel_trim_bank1": [0.0],
        }
    )
    f = ved_frame(df, max_torque_nm=150.0)

    assert f["efficiency"].iloc[0] == pytest.approx(0.30, abs=0.001)
    assert f["load"].iloc[0] == 50.0


def test_impossible_efficiencies_and_rows_with_missing_inputs_are_dropped_by_build_xy():
    f = pd.DataFrame(
        {
            "engine_speed": [2000.0] * 4, "load": [40.0, 40.0, np.nan, 40.0], "vehicle_speed": [50.0] * 4,
            "rpm_per_kmh": [40.0] * 4, "total_fuel_trim": [0.0] * 4, "efficiency": [0.3, 0.9, 0.3, np.nan],
        }
    )
    X, y = build_xy(f)

    assert X.shape == (1, 5) and y.tolist() == [0.3]  # 0.9 is above what a petrol engine can do; the others are incomplete


def test_group_split_keeps_every_group_wholly_on_one_side():
    groups = pd.Series(list("AABBCCDD"))
    train, test = group_split(groups, held_out=["B", "D"])

    assert train.tolist() == [True, True, False, False, True, True, False, False]
    assert (train ^ test).all()


def data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    return X, 0.2 + 0.05 * X[:, 0] + 0.02 * X[:, 1] ** 2  # efficiency depends on features 0 and 1 only


@pytest.mark.parametrize("fit", [fit_gbm_efficiency, fit_ann_efficiency])
def test_each_model_learns_a_smooth_relationship_and_predicts_it_on_unseen_rows(fit):
    X, y = data()
    Xt, yt = data(seed=1)
    r = evaluate_regression(yt, fit(X, y).predict(Xt))

    assert r["r2"] > 0.9 and r["mae"] < 0.01


@pytest.mark.parametrize("fit", [fit_gbm_efficiency, fit_ann_efficiency])
def test_the_same_seed_gives_the_same_predictions(fit):
    X, y = data(400)

    assert np.array_equal(fit(X, y, seed=2).predict(X), fit(X, y, seed=2).predict(X))


def test_permutation_importance_ranks_the_features_the_target_depends_on_first():
    X, y = data()
    Xt, yt = data(seed=1)
    imp = feature_importance(fit_gbm_efficiency(X, y), Xt, yt, names=list("abcde"))

    assert list(imp)[:2] == ["a", "b"]  # feature 0 moves y by 0.05 per sd, feature 1 by up to 0.02 x x^2
    assert imp["a"] > 5 * imp["c"] and imp["c"] < 0.001  # an irrelevant feature costs nothing when shuffled


def test_regression_scores_match_a_hand_calculation():
    # true 1,2,3 predicted 1,2,4: MAE = 1/3; SSE = 1, SST = 2 so R2 = 1 - 1/2 = 0.5.
    r = evaluate_regression([1, 2, 3], [1, 2, 4])

    assert r["mae"] == pytest.approx(1 / 3) and r["r2"] == pytest.approx(0.5)
