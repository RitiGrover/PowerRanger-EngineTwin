from pathlib import Path

import numpy as np
import pytest

from enginetwin.ocslab import load_ocslab
from enginetwin.simulator import simulate_window

CSVS = sorted((Path(__file__).resolve().parents[1] / "data" / "osclab").glob("*.csv"))

pytestmark = pytest.mark.skipif(not CSVS, reason="Ocslab CSV not downloaded")

# simulated OBD name -> real Ocslab column
SIGNALS = {
    "calculated_load_value": "calculated_load_value",
    "engine_coolant_temperature": "engine_coolant_temperature",
    "timing_advance": "current_spark_timing",
    "short_term_fuel_trim_bank1": "short_term_fuel_trim_bank1",
    "long_term_fuel_trim_bank1": "long_term_fuel_trim_bank1",
    "engine_speed": "engine_speed",
}
# rpm and load were used to pick the real rows. Their spread there is just the width of the bin, not physics.
SPREAD_IS_PHYSICAL = {"engine_coolant_temperature", "timing_advance", "short_term_fuel_trim_bank1", "long_term_fuel_trim_bank1"}


@pytest.fixture(scope="module")
def real_and_simulated():
    d = load_ocslab(CSVS[0])
    real = d[d["engine_speed"].between(1800, 2200) & d["calculated_load_value"].between(35, 45) & (d["vehicle_speed"] > 20)]
    sim = [simulate_window(rpm=2000, load=0.4, seed=s).obd for s in range(300)]
    return real, sim


@pytest.mark.parametrize("name", SIGNALS)
def test_simulated_healthy_obd_averages_match_the_real_car_at_the_same_operating_point(real_and_simulated, name):
    real, sim = real_and_simulated
    real_col = real[SIGNALS[name]]

    sim_mean = np.mean([o[name] for o in sim])

    assert abs(sim_mean - real_col.mean()) <= max(1.0, 0.5 * real_col.std())


@pytest.mark.parametrize("name", sorted(SPREAD_IS_PHYSICAL))
def test_simulated_healthy_obd_varies_about_as_much_as_the_real_car(real_and_simulated, name):
    real, sim = real_and_simulated

    ratio = np.std([o[name] for o in sim]) / real[SIGNALS[name]].std()

    assert 0.5 <= ratio <= 2.0
