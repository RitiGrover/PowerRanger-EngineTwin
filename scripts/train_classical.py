"""Fit the SVM and softmax regression on the hand-made features and score them on every split."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pickle
import time

import numpy as np

from common import CACHE, cache_file, load_splits
from enginetwin.classifiers import fit_softmax, fit_svm, top_features
from enginetwin.evaluation import evaluate
from enginetwin.fault_features import FEATURE_NAMES, feature_matrix
from enginetwin.schema import FAULTS

splits = load_splits()
X = {}
for name, s in zip(splits._fields, splits):
    t = time.time()
    X[name] = feature_matrix(s, cache_file("features", name))
    print(f"features {name}: {X[name].shape} in {time.time() - t:.0f}s", flush=True)

results = {}
for model_name, fit in (("svm", fit_svm), ("softmax", fit_softmax)):
    model = fit(X["train"], splits.train.fault)
    with open(CACHE / f"model_{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)
    results[model_name] = {}
    for name, s in zip(splits._fields, splits):
        r = evaluate(s.fault, model.predict(X[name]), list(FAULTS))
        results[model_name][name] = {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "confusion": r["confusion"].tolist()}
        print(f"{model_name:8s} {name:18s} acc {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f}", flush=True)
    if model_name == "softmax":
        results["softmax_top_features"] = {
            label: top_features(model, FEATURE_NAMES, label, n=5) for label in FAULTS if label != "healthy"
        }
(CACHE / "results_classical.json").write_text(json.dumps(results, indent=1))
