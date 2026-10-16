from pathlib import Path

import pandas as pd

from enginetwin.liu_guard import SCENARIOS, build_guard_replay, load_guard_bundle


def test_prepared_guard_bundle_has_three_distinct_held_out_replays():
    path = Path(__file__).parents[1] / "data" / "liu_ice" / "guard_bundle.joblib"
    if not path.exists():
        return
    bundle = load_guard_bundle(path)
    assert set(bundle["test_windows"]["trajectory_id"].unique()) == set(SCENARIOS)
    for scenario_id in SCENARIOS:
        replay = build_guard_replay(bundle, scenario_id)
        assert len(replay.frame) > 20
        assert replay.evidence.shape[1] == 3
        assert replay.health.between(18, 100).all()
        assert isinstance(replay.sensor_readings(0), list)
        assert len(replay.sensor_readings(0)) == 10


def test_guard_confirmation_requires_sustained_evidence():
    path = Path(__file__).parents[1] / "data" / "liu_ice" / "guard_bundle.joblib"
    if not path.exists():
        return
    replay = build_guard_replay(load_guard_bundle(path), "wltp_f_pim_105")
    first = replay.confirmed.any(axis=1).idxmax()
    assert not replay.confirmed.iloc[: max(0, first - 1)].to_numpy().any()
    assert replay.evidence.iloc[first].max() >= 1.25
    assert pd.notna(replay.injection_time)
