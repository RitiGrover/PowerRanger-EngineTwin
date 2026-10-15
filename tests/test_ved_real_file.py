from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from enginetwin.features import add_instant_features, trip_features
from enginetwin.ved import load_ved

VED = Path(__file__).resolve().parents[1] / "data" / "ved"
WEEKS = sorted((VED / "extracted").glob("*.csv"))[:2]
STATIC = VED / "VED_Static_Data_ICE&HEV.xlsx"

pytestmark = pytest.mark.skipif(not WEEKS or not STATIC.exists(), reason="VED not downloaded and extracted")


@pytest.fixture(scope="module")
def ved():
    return load_ved(WEEKS, static_path=STATIC)


def test_only_combustion_cars_remain(ved):
    static = pd.read_excel(STATIC)
    ice_ids = set(static.loc[static["Vehicle Type"] == "ICE", "VehId"])
    assert set(ved["vehid"]) <= ice_ids


def test_most_samples_have_a_usable_fuel_flow(ved):
    assert ved["fuel_consumption"].notna().mean() > 0.5  # from MAF + trims where no fuel rate is reported


def test_trip_fuel_economy_is_plausible_for_a_mixed_fleet(ved):
    feats = trip_features(add_instant_features(ved))
    duration = ved.groupby("trip_id")["time_s"].max()
    long_trips = feats[feats["trip_id"].map(duration) >= 300]

    economy = long_trips["fuel_economy_km_per_l"].dropna()

    assert len(economy) > 100
    assert not np.isinf(feats["fuel_economy_km_per_l"]).any()
    assert 6 <= economy.median() <= 18  # roughly 14-42 mpg; no mixed fleet averages outside this
