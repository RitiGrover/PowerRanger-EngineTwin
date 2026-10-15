"""Fit the OBD-only network and score it on every split."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, load_splits
from enginetwin.evaluation import evaluate
from enginetwin.obd_ann import fit_obd_ann, obd_matrix
from enginetwin.schema import FAULTS

splits = load_splits()
model = fit_obd_ann(obd_matrix(splits.train), splits.train.fault)
with open(CACHE / "model_obd_ann.pkl", "wb") as f:
    pickle.dump(model, f)

results = {}
for name, s in zip(splits._fields, splits):
    predicted = model.predict(obd_matrix(s))
    r = evaluate(s.fault, predicted, list(FAULTS))
    results[name] = {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "confusion": r["confusion"].tolist()}
    print(f"obd_ann  {name:18s} acc {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f}", flush=True)
test = splits.test_same
predicted = model.predict(obd_matrix(test))
for family in ("misfire", "knock", "valve_wear"):
    flagged = np.array([family in p for p in predicted])
    present = np.array([family in f for f in test.fault])
    results[f"{family}_family_detection"] = {"found": float(flagged[present].mean()), "false_alarm": float(flagged[~present].mean())}
    print(f"{family:11s} found {flagged[present].mean():.2f}  false alarm {flagged[~present].mean():.2f}")
(CACHE / "results_obd_ann.json").write_text(json.dumps(results, indent=1))
