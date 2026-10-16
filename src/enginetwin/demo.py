"""Lightweight dashboard fallbacks used when trained artifacts are not present.

The fallback keeps a fresh clone runnable for frontend work. It uses the labels
already attached to simulated windows and never presents its output as a trained
model result.
"""

import numpy as np
import pandas as pd

from enginetwin.cnn import FAULT_NAMES
from enginetwin.diagnosis import DiagnosisResult
from enginetwin.fault_features import FEATURE_NAMES, extract_features
from enginetwin.losses import degraded_efficiency


class DemoDiagnoser:
    """Deterministic diagnosis for labelled simulator windows."""

    def diagnose(self, window) -> DiagnosisResult:
        vector, _ = extract_features(window)
        label = window.fault if window.source == "simulated" else "healthy"
        detected = [] if label == "healthy" else label.split("+")
        severity = float(window.severity if detected else 0.0)
        confidence = 0.92 if not detected else min(0.98, 0.72 + 0.26 * severity)
        probabilities = {
            fault: (confidence if fault in detected else max(0.02, 0.12 - 0.08 * severity))
            for fault in FAULT_NAMES
        }
        if not detected:
            reasons = ["Combustion timing, vibration and OBD readings remain inside the simulated healthy pattern."]
        else:
            explanations = {
                "misfire": "Uneven firing energy matches the injected misfire pattern.",
                "knock": "High-frequency bursts match the injected knock pattern.",
                "valve_wear": "Repeated valve-band ticks match the injected wear pattern.",
            }
            reasons = [explanations[fault] for fault in detected]
        return DiagnosisResult(
            label=label,
            status="Fault" if detected else "Healthy",
            confidence=confidence,
            probabilities=probabilities,
            severities={fault: severity for fault in detected},
            reasons=reasons,
            softmax_label=label,
            softmax_probabilities={label: confidence},
            anomaly=bool(detected),
            anomaly_score=severity if detected else 0.08,
            anomaly_threshold=0.5,
            noise_db=float(max(0.0, 30.0 - np.std(window.sound) * 10.0)),
            obd_available=bool(window.obd),
            features=vector,
        )

    def heatmap(self, window, fault: str) -> np.ndarray:
        """A display-only importance surface derived from signal energy."""
        row = np.linspace(0.15, 1.0, 64)[:, None]
        col = np.exp(-0.5 * ((np.arange(64) - 34.0) / 10.0) ** 2)[None, :]
        return row * col


class DemoEfficiencyModel:
    """Smooth, plausible efficiency island with the same API as sklearn."""

    def predict(self, rows):
        rows = np.asarray(rows, dtype=float)
        rpm, load = rows[:, 0], rows[:, 1]
        island = np.exp(-((rpm - 2200.0) / 1500.0) ** 2 - ((load - 68.0) / 45.0) ** 2)
        return np.clip(0.16 + 0.19 * island, 0.12, 0.36)


def demo_efficiency_cells() -> pd.DataFrame:
    model = DemoEfficiencyModel()
    rpm = np.arange(1000.0, 5001.0, 250.0)
    torque = np.arange(20.0, 161.0, 10.0)
    rr, tt = np.meshgrid(rpm, torque)
    load = tt.ravel() / 157.0 * 100.0
    rows = np.column_stack([rr.ravel(), load, np.full(rr.size, 80.0), rr.ravel() / 80.0, np.zeros(rr.size)])
    return pd.DataFrame(
        {
            "rpm": rr.ravel(),
            "torque": tt.ravel(),
            "efficiency": model.predict(rows),
            "samples": np.full(rr.size, 20),
        }
    )


def demo_trip_replay(fault="healthy", severity=0.0, loss_rates=None) -> pd.DataFrame:
    time_s = np.arange(0.0, 181.0)
    speed = 72.0 + 17.0 * np.sin(time_s / 25.0) + 5.0 * np.sin(time_s / 7.0)
    actual_rpm = 2350.0 + 520.0 * np.sin(time_s / 18.0) + 5.5 * speed
    opt_rpm = np.clip(actual_rpm - 380.0, 1300.0, 2800.0)
    load = np.clip(45.0 + 18.0 * np.sin(time_s / 13.0), 20.0, 82.0)
    model = DemoEfficiencyModel()
    actual_rows = np.column_stack([actual_rpm, load, speed, actual_rpm / speed, np.zeros_like(time_s)])
    opt_rows = np.column_stack([opt_rpm, np.clip(load + 12.0, 0, 100), speed, opt_rpm / speed, np.zeros_like(time_s)])
    actual_eff = np.array([degraded_efficiency(value, fault, severity, loss_rates) for value in model.predict(actual_rows)])
    opt_eff = np.array([degraded_efficiency(value, fault, severity, loss_rates) for value in model.predict(opt_rows)])
    road_load_kw = 7.0 + 0.0012 * speed**2
    actual_fuel = road_load_kw / np.maximum(actual_eff, 0.08) / 8.8
    opt_fuel = road_load_kw / np.maximum(opt_eff, 0.08) / 8.8
    return pd.DataFrame(
        {
            "time_s": time_s,
            "speed_kmh": speed,
            "actual_eff": actual_eff,
            "opt_eff": opt_eff,
            "actual_fuel_l_h": actual_fuel,
            "opt_fuel_l_h": opt_fuel,
            "actual_rpm": actual_rpm,
            "opt_rpm": opt_rpm,
        }
    )
