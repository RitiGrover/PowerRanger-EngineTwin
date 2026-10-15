"""Fit the severity models and the healthy baseline, and bundle everything the dashboard's diagnoser needs in cache/."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, cache_file, load_splits
from enginetwin.diagnosis import Diagnoser
from enginetwin.explain import healthy_baseline
from enginetwin.fault_features import feature_matrix
from enginetwin.severity import fit_severity_models

splits = load_splits()
X = feature_matrix(splits.train, cache_file("features", "train"))
load = lambda name: pickle.load(open(CACHE / name, "rb"))
autoencoder = load("model_autoencoder.pkl")
Diagnoser(
    cnn=load("model_cnn.pkl"),
    softmax=load("model_softmax.pkl"),
    autoencoder=autoencoder["model"],
    thresholds=autoencoder["thresholds"],
    severity_models=fit_severity_models(X, splits.train.fault, splits.train.severity),
    baseline=healthy_baseline(X[splits.train.fault == "healthy"]),
).save(CACHE)
print("saved", [p.name for p in CACHE.glob("*.pkl")])
