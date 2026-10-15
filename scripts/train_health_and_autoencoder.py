"""Fit the healthy-baseline autoencoder and the valve-wear health index, and score them."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, cache_file, load_splits
from enginetwin.autoencoder import anomaly_auroc, fit_autoencoder, healthy_threshold
from enginetwin.noise import estimate_noise_db, fit_noise_thresholds
from enginetwin.fault_features import extract_features, feature_matrix
from enginetwin.health import fit_severity_regressor, health_index, time_to_service
from enginetwin.schema import FAULTS
from enginetwin.wear import wear_series

splits = load_splits()
X = {name: feature_matrix(s, cache_file("features", name)) for name, s in zip(splits._fields, splits)}

# Autoencoder: healthy training windows only, with an alarm level per estimated noise level
EDGES = (-22.5, -17.5, -12.5, -7.5)  # noise-to-signal dB; the wide validation set (SNR 5-25 dB) has healthy windows in every bin
ae = fit_autoencoder(X["train"][splits.train.fault == "healthy"], seed=0)
noise_db = {name: np.array([estimate_noise_db(s.window(i)) for i in range(len(s))]) for name, s in zip(splits._fields, splits)}
wide = splits.validation_wide
single = healthy_threshold(ae, X["validation_wide"], wide.fault, q=0.99)  # one level for every noise level (for comparison)
thresholds = fit_noise_thresholds(ae.score(X["validation_wide"]), noise_db["validation_wide"], wide.fault, EDGES, q=0.99)
with open(CACHE / "model_autoencoder.pkl", "wb") as f:
    pickle.dump({"model": ae, "thresholds": thresholds, "single_threshold": single}, f)
results = {"autoencoder": {"edges": list(EDGES), "bin_thresholds": thresholds.thresholds.tolist()}, "autoencoder_single_threshold": {"threshold": single}}
for name, s in zip(splits._fields, splits):
    scores = ae.score(X[name])
    limit = thresholds.threshold_for(noise_db[name])
    healthy = scores[s.fault == "healthy"]
    per_fault = {
        fault: {
            "auroc": anomaly_auroc(healthy, scores[s.fault == fault]),
            "flagged": float(np.mean(scores[s.fault == fault] > limit[s.fault == fault])),
            "flagged_single_threshold": float(np.mean(scores[s.fault == fault] > single)),
        }
        for fault in FAULTS if fault != "healthy"
    }
    results["autoencoder"][name] = {
        "false_alarm": float(np.mean(healthy > limit[s.fault == "healthy"])),
        "false_alarm_single_threshold": float(np.mean(healthy > single)),
        "mean_estimated_noise_db": float(noise_db[name].mean()),
        "per_fault": per_fault,
    }
    r = results["autoencoder"][name]
    print(f"AE {name:16s} noise {r['mean_estimated_noise_db']:6.1f} dB  false alarm {r['false_alarm']:.3f} (single {r['false_alarm_single_threshold']:.2f})  flagged "
          + " ".join(f"{k[:11]}:{v['flagged']:.2f}" for k, v in per_fault.items()), flush=True)

# Valve wear health index
target = lambda s: np.array([sev if "valve_wear" in f else 0.0 for f, sev in zip(s.fault, s.severity)])
regressor = fit_severity_regressor(X["train"], target(splits.train))
with open(CACHE / "model_wear.pkl", "wb") as f:
    pickle.dump(regressor, f)
err = np.abs(regressor.predict(X["test_same"]) - target(splits.test_same))
results["wear_regressor_mae_test_same"] = float(err.mean())
print(f"wear estimate MAE on test_same: {err.mean():.3f}")

series = wear_series(n_days=60, seed=5, rpm=2000.0, load=0.4)
estimated = regressor.predict(np.array([extract_features(w)[0] for w in series.windows]))
true_severity = np.array([w.severity for w in series.windows])
health = health_index(estimated)
crossing_day = int(np.argmax(true_severity >= 0.8))  # health 20 when severity is 0.8
forecasts = {}
for day in (20, 30, 40, 50):
    ahead = time_to_service(series.days[: day + 1], health[: day + 1], threshold=20, last=14)
    forecasts[day] = {"predicted_service_day": float(day + ahead), "true_service_day": crossing_day}
    print(f"on day {day}: service predicted on day {day + ahead:.1f}, true day {crossing_day}")
results["wear_series"] = {
    "days": series.days.tolist(), "true_health": health_index(true_severity).tolist(), "estimated_health": health.tolist(),
    "forecasts": forecasts,
}
(CACHE / "results_health_autoencoder.json").write_text(json.dumps(results, indent=1))
