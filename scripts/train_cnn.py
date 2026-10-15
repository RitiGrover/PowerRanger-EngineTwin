"""Fit the multimodal CNN (spectrograms + OBD) and score it on every split."""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, cache_file, load_splits
from enginetwin.cnn import faults_to_multihot, fit_cnn, labels_from_proba
from enginetwin.evaluation import evaluate
from enginetwin.obd_ann import obd_matrix
from enginetwin.schema import FAULTS
from enginetwin.spectrograms import spectrogram_stack

splits = load_splits()
spec = {}
for name, s in zip(splits._fields, splits):
    t = time.time()
    spec[name] = spectrogram_stack(s, cache_file("spectrograms", name))
    print(f"spectrograms {name}: {spec[name].shape} in {time.time() - t:.0f}s", flush=True)

t = time.time()
model = fit_cnn(spec["train"], obd_matrix(splits.train), faults_to_multihot(splits.train.fault), seed=0, epochs=20)
print(f"trained in {time.time() - t:.0f}s, {model.n_parameters} parameters", flush=True)
with open(CACHE / "model_cnn.pkl", "wb") as f:
    pickle.dump(model, f)

results = {}
for name, s in zip(splits._fields, splits):
    proba = model.predict_proba(spec[name], obd_matrix(s))
    r = evaluate(s.fault, labels_from_proba(proba), list(FAULTS))
    flags = proba > 0.5
    truth = faults_to_multihot(s.fault) == 1
    results[name] = {
        "accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "confusion": r["confusion"].tolist(),
        "per_fault_recall": dict(zip(("misfire", "knock", "valve_wear"), (flags[truth[:, k], k].mean() for k in range(3)))),
        "per_fault_false_alarm": dict(zip(("misfire", "knock", "valve_wear"), (flags[~truth[:, k], k].mean() for k in range(3)))),
    }
    results[name]["per_fault_recall"] = {k: float(v) for k, v in results[name]["per_fault_recall"].items()}
    results[name]["per_fault_false_alarm"] = {k: float(v) for k, v in results[name]["per_fault_false_alarm"].items()}
    print(f"cnn {name:18s} acc {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f}", flush=True)
(CACHE / "results_cnn.json").write_text(json.dumps(results, indent=1))
