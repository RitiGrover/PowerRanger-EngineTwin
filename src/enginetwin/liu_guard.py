"""Adapter for the measured LiU-ICE held-out sensor-consistency replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from enginetwin.liu_ice import sensor_consistency_evidence


SCENARIOS = {
    "wltp_f_pim_105": {
        "name": "Manifold pressure drift",
        "code": "PIM +5%",
        "fault_sensor": "intake_manifold_pressure",
    },
    "ftp75city2_f_pic_115": {
        "name": "Intercooler pressure drift",
        "code": "PIC +15%",
        "fault_sensor": "Intercooler_pressure",
    },
    "wltp_f_waf_115": {
        "name": "Air mass flow over-read",
        "code": "WAF +15%",
        "fault_sensor": "air_mass_flow",
    },
}

SENSORS = {
    "Intercooler_pressure": ("Intercooler pressure", "kPa", lambda value: value / 1000),
    "intercooler_temperature": ("Intercooler outlet", "°C", lambda value: value - 273.15),
    "intake_manifold_pressure": ("Manifold pressure", "kPa", lambda value: value / 1000),
    "air_mass_flow": ("Air mass flow", "g/s", lambda value: value * 1000),
    "engine_speed": ("Engine speed", "RPM", lambda value: value * 60 / (2 * np.pi)),
    "throttle_position": ("Throttle", "deg", float),
    "wastegate_position": ("Wastegate", "%", lambda value: value * 100),
    "injected_fuel_mass": ("Injected fuel", "g/s", lambda value: value * 1000),
    "ambient_pressure": ("Ambient pressure", "kPa", lambda value: value / 1000),
    "ambient_temperature": ("Ambient temperature", "°C", lambda value: value - 273.15),
}

DETECTOR_BY_SENSOR = {
    "Intercooler_pressure": "f_pic",
    "intake_manifold_pressure": "f_pim",
    "air_mass_flow": "f_waf",
}


@dataclass(frozen=True)
class GuardReplay:
    scenario_id: str
    frame: pd.DataFrame
    evidence: pd.DataFrame
    confirmed: pd.DataFrame
    health: pd.Series

    @property
    def scenario(self) -> dict:
        return SCENARIOS[self.scenario_id]

    @property
    def injection_time(self) -> float:
        return float(self.frame["injection_time_s"].iloc[0])

    def cursor_for_progress(self, progress: float) -> int:
        return min(len(self.frame) - 1, max(0, int(round((len(self.frame) - 1) * progress))))

    def sensor_readings(self, cursor: int) -> list[tuple[str, str, str, str, str]]:
        row = self.frame.iloc[cursor]
        pre_fault = self.frame.loc[self.frame["window_end_s"] < self.injection_time]
        readings = []
        for sensor, (name, unit, convert) in SENSORS.items():
            raw = float(row[f"{sensor}__mean"])
            value = float(convert(raw))
            baseline = pre_fault[f"{sensor}__mean"]
            scale = max(float(baseline.std()), abs(float(baseline.mean())) * 0.005, 1e-9)
            deviation = abs(raw - float(baseline.median())) / scale
            detector = DETECTOR_BY_SENSOR.get(sensor)
            is_confirmed = bool(detector and self.confirmed.iloc[cursor][detector])
            state = "FAULT" if is_confirmed else ("WATCH" if deviation >= 2 else "MEASURED")
            colour = "#ff5d5d" if is_confirmed else ("#f4c95d" if deviation >= 2 else "#d8aa32")
            formatted = f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.1f}"
            readings.append((name, formatted, unit, state, colour))
        return readings


def load_guard_bundle(path: str | Path) -> dict:
    bundle = joblib.load(Path(path))
    required = {"consistency", "test_windows"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"LiU-ICE guard bundle is missing: {', '.join(sorted(missing))}")
    return bundle


def build_guard_replay(bundle: dict, scenario_id: str) -> GuardReplay:
    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown LiU-ICE scenario: {scenario_id}")
    frame = bundle["test_windows"].loc[
        bundle["test_windows"]["trajectory_id"].eq(scenario_id)
    ].sort_values("window_end_s").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"Held-out trajectory {scenario_id} is absent from the guard bundle.")
    evidence = sensor_consistency_evidence(bundle["consistency"], frame)
    confirmed = pd.DataFrame(False, index=frame.index, columns=evidence.columns)
    for code in evidence:
        hits = evidence[code].ge(1.0).astype(int)
        confirmed[code] = hits.rolling(3, min_periods=1).sum().ge(2) & evidence[code].ge(1.25)
    peak = evidence.max(axis=1)
    health = (100 - 62 * np.clip((peak - 0.5) / 1.5, 0, 1)).clip(18, 100)
    return GuardReplay(scenario_id, frame, evidence, confirmed, health)
