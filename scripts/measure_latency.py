"""Alert latency of each model on simulated fault onsets, and the combined results table."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE
from enginetwin.latency import alert_latency, predict_with_cnn, predict_with_features, predict_with_obd, simulate_onset
from enginetwin.results import build_results_table

load = lambda name: pickle.load(open(CACHE / f"model_{name}.pkl", "rb"))
predictors = {
    "svm": predict_with_features(load("svm")),
    "softmax": predict_with_features(load("softmax")),
    "obd_ann": predict_with_obd(load("obd_ann")),
    "cnn": predict_with_cnn(load("cnn")),
}
CONFIRM = int(sys.argv[1]) if len(sys.argv) > 1 else 2  # consecutive alerts needed
SUFFIX = "" if CONFIRM == 2 else f"_confirm{CONFIRM}"
FAULTS = ("misfire", "knock", "valve_wear")
RPMS = (1500.0, 2400.0, 3000.0)
SEEDS = (101, 102, 103)  # not used in any training split

results = {}
for name, predict in predictors.items():
    results[name] = {}
    for fault in FAULTS:
        runs = [
            alert_latency(predict, simulate_onset(fault, 0.8, onset_s=3.0, duration_s=7.0, rpm=rpm, seed=seed), confirm=CONFIRM)
            for rpm in RPMS for seed in SEEDS
        ]
        found = [r["latency_s"] for r in runs if r["latency_s"] is not None]
        results[name][fault] = {
            "detected": len(found) / len(runs),
            "mean_latency_s": float(np.mean(found)) if found else None,
            "mean_compute_s": float(np.mean([r["compute_s"] for r in runs if r["compute_s"] is not None])) if found else None,
            "false_alarm_rate": float(np.mean([r["false_alarm_rate"] for r in runs])),
        }
        print(name, fault, results[name][fault], flush=True)
(CACHE / f"results_latency{SUFFIX}.json").write_text(json.dumps(results, indent=1))
table = build_results_table(CACHE)
table.to_csv(CACHE / "results_table.csv", index=False)
print(table.pivot(index="model", columns="split", values="accuracy").round(3))
