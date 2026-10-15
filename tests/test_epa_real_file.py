from pathlib import Path

import pytest

from enginetwin.efficiency_map import efficiency_map, relative_sweet_spot, sweet_spot
from enginetwin.epa import load_epa_map
from enginetwin.ocslab import load_ocslab

ROOT = Path(__file__).resolve().parents[1]
EPA = sorted((ROOT / "data" / "epa" / "extracted").glob("6-*Fuel Map Data.xlsx"))
OCSLAB = sorted((ROOT / "data" / "osclab").glob("*.csv"))

pytestmark = pytest.mark.skipif(not EPA, reason="EPA ALPHA map not downloaded")


@pytest.fixture(scope="module")
def epa():
    return load_epa_map(EPA[0])


def test_epa_map_covers_the_engines_operating_range(epa):
    assert len(epa) > 200
    assert epa["rpm"].min() < 1000 and epa["rpm"].max() > 5000
    assert epa["efficiency"].between(0, 0.5).all()


def test_epa_peak_efficiency_matches_a_modern_high_compression_petrol_engine(epa):
    assert 0.30 <= sweet_spot(epa)["efficiency"] <= 0.42


@pytest.mark.skipif(not OCSLAB, reason="Ocslab CSV not downloaded")
def test_ocslab_and_epa_sweet_spots_sit_in_the_same_region_of_their_own_maps(epa):
    ocslab = efficiency_map(load_ocslab(OCSLAB[0]))

    epa_rpm, epa_torque = relative_sweet_spot(epa)
    oc_rpm, oc_torque = relative_sweet_spot(ocslab)

    # Different engines. Compare position within each engine's own range, not absolute values.
    assert abs(epa_rpm - oc_rpm) < 0.25
    assert abs(epa_torque - oc_torque) < 0.25
