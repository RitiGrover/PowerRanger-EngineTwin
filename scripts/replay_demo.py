"""Replay one held-out Ocslab trip and one VED trip: actual against optimised, written to cache/ for the dashboard."""
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, ROOT
from enginetwin.features import add_instant_features
from enginetwin.ocslab import load_ocslab
from enginetwin.replay import replay_trip
from enginetwin.vehicle import VehicleSpec
from enginetwin.ved import load_ved

model = pickle.load(open(CACHE / "model_efficiency_gbm.pkl", "rb"))  # trained on Ocslab drivers other than C and H
spec = VehicleSpec()
summaries = {}

oc = load_ocslab(next((ROOT / "data" / "osclab").glob("*.csv")))
held = oc[oc["driver"].isin(["C", "H"])]
trip_id = held.groupby("trip_id").size().sort_values().index[-1]  # the longest held-out trip
table, summaries["ocslab"] = replay_trip(held[held.trip_id == trip_id], model, spec, "ocslab")
table.to_csv(CACHE / "replay_ocslab.csv", index=False)
summaries["ocslab"]["trip_id"] = trip_id
print("ocslab", trip_id, len(table), summaries["ocslab"], flush=True)

ved = load_ved([sorted((ROOT / "data" / "ved" / "extracted").glob("*.csv"))[0]], ROOT / "data" / "ved" / "VED_Static_Data_ICE&HEV.xlsx")
ved = ved.dropna(subset=["vehicle_speed", "engine_speed", "fuel_consumption"])
sizes = ved.groupby("trip_id").size()
trip_id = sizes[(sizes > 500) & (sizes < 1500)].sort_values().index[-1]
table, summaries["ved"] = replay_trip(ved[ved.trip_id == trip_id].sort_values("time_s"), model, spec, "ved")
table.to_csv(CACHE / "replay_ved.csv", index=False)
summaries["ved"]["trip_id"] = trip_id
print("ved", trip_id, len(table), summaries["ved"], flush=True)
(CACHE / "replay_summary.json").write_text(json.dumps(summaries, indent=1))
