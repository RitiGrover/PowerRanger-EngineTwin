"""Generate (or reuse) the cached labelled simulated dataset: five splits of 2 s sound, vibration and OBD windows.

Healthy efficiency comes from the real Ocslab efficiency map; fault losses are the assumed rates in enginetwin.losses.
"""
from pathlib import Path

from enginetwin.efficiency_map import efficiency_map
from enginetwin.labelled import cached_labelled_splits, healthy_efficiency_surface
from enginetwin.ocslab import load_ocslab

N_TRAIN, N_VAL, N_TEST, SEED = 300, 100, 100, 2024  # windows per fault class

root = Path(__file__).resolve().parents[1]
cells = efficiency_map(load_ocslab(next((root / "data" / "osclab").glob("*.csv"))))
surface = healthy_efficiency_surface(cells, max_torque_nm=cells["torque"].max())
splits = cached_labelled_splits(root / "cache", N_TRAIN, N_VAL, N_TEST, SEED, surface)
for name, s in zip(splits._fields, splits):
    print(f"{name:18s} {len(s):5d} windows")
