"""Fit the efficiency models on Ocslab (whole drivers held out) and test them on held-out Ocslab drivers and on VED cars."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, ROOT
from enginetwin.efficiency_model import (
    FEATURES, KIA_SOUL_MAX_TORQUE_NM, build_xy, evaluate_regression, feature_importance, fit_ann_efficiency,
    fit_gbm_efficiency, group_split, ocslab_frame, ved_frame,
)
from enginetwin.ocslab import load_ocslab
from enginetwin.ved import load_ved

HELD_OUT_DRIVERS = ["C", "H"]
VED_WEEKS = 3

oc = ocslab_frame(__import__("enginetwin.features", fromlist=["x"]).add_instant_features(load_ocslab(next((ROOT / "data" / "osclab").glob("*.csv")))))
train_mask, test_mask = group_split(oc["group"], HELD_OUT_DRIVERS)
X_train, y_train = build_xy(oc[train_mask])
X_test, y_test = build_xy(oc[test_mask])

weeks = sorted((ROOT / "data" / "ved" / "extracted").glob("*.csv"))[:VED_WEEKS]
ved = ved_frame(load_ved(weeks, ROOT / "data" / "ved" / "VED_Static_Data_ICE&HEV.xlsx"), KIA_SOUL_MAX_TORQUE_NM)
X_ved, y_ved = build_xy(ved)
print(f"Ocslab train {len(y_train)} rows (8 drivers), test {len(y_test)} rows (drivers {HELD_OUT_DRIVERS}); VED {len(y_ved)} rows, {ved['group'].nunique()} cars")
print(f"mean efficiency: Ocslab train {y_train.mean():.3f}, Ocslab test {y_test.mean():.3f}, VED (assumed torque) {y_ved.mean():.3f}")

results = {"baseline_mean_mae": {"ocslab_held_out": float(np.mean(np.abs(y_test - y_train.mean()))), "ved": float(np.mean(np.abs(y_ved - y_train.mean())))}}
for name, fit in (("gbm", fit_gbm_efficiency), ("ann", fit_ann_efficiency)):
    model = fit(X_train, y_train)
    with open(CACHE / f"model_efficiency_{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    results[name] = {
        "ocslab_held_out": evaluate_regression(y_test, model.predict(X_test)),
        "ved": evaluate_regression(y_ved, model.predict(X_ved)),
    }
    results[name]["importance_ocslab_held_out"] = feature_importance(model, X_test, y_test, FEATURES)
    print(name, json.dumps(results[name], indent=1))
(CACHE / "results_efficiency.json").write_text(json.dumps(results, indent=1))
