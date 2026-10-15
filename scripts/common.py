"""Shared set-up for the training scripts: project paths and the cached labelled dataset."""
from pathlib import Path

from enginetwin.efficiency_map import efficiency_map
from enginetwin.labelled import cache_key, cached_labelled_splits, healthy_efficiency_surface
from enginetwin.ocslab import load_ocslab

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
N_TRAIN, N_VAL, N_TEST, SEED = 300, 100, 100, 2024  # windows per fault class


def _surface():
    cells = efficiency_map(load_ocslab(next((ROOT / "data" / "osclab").glob("*.csv"))))
    return healthy_efficiency_surface(cells, max_torque_nm=cells["torque"].max())


def load_splits():
    return cached_labelled_splits(CACHE, N_TRAIN, N_VAL, N_TEST, SEED, _surface())


def cache_file(kind, split, suffix="npy"):
    """Per-split cache path tied to the dataset's key, so a regenerated dataset can never reuse stale features."""
    return CACHE / f"{kind}_{cache_key(N_TRAIN, N_VAL, N_TEST, SEED, _surface())}_{split}.{suffix}"
