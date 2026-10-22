"""Data behind the Engine map: what each sensor reads, and what the diagnosis does with it, for every fault.

Everything is computed with the project's existing simulator, feature extractor and diagnoser. Nothing here
changes a detection step; it only runs the same calls once per fault label and packages the numbers for display."""
import numpy as np

from enginetwin.fault_features import FEATURE_NAMES, extract_features
from enginetwin.schema import FAULTS
from enginetwin.simulator import simulate_window

# The features the map shows. Each one comes from one physical sensor channel.
SHOWN_FEATURES = (
    "sound_half_order_ratio", "sound_knock_vs_floor", "vibration_cyl_cv", "vibration_cyl_min_ratio",
    "vibration_knock_vs_floor", "vibration_valve_vs_floor", "vibration_valve_env_cycle",
    "obd_engine_speed", "obd_total_fuel_trim", "obd_timing_advance", "obd_maf",
    "obd_calculated_load_value", "obd_engine_coolant_temperature",
)
WAVE_BINS = 800        # 2 s of 16 kHz audio drawn as min/max pairs: narrow knock bursts survive the reduction
HEALTHY_SEEDS = 8      # healthy windows used to learn how much each feature normally wobbles


def _envelope(signal: np.ndarray, scale: float) -> list:
    n = (len(signal) // WAVE_BINS) * WAVE_BINS
    blocks = signal[:n].reshape(WAVE_BINS, -1) / scale
    return np.round(np.stack([blocks.min(axis=1), blocks.max(axis=1)], axis=1), 3).tolist()


def build_payload(diagnose, rpm: float, load: float, severity: float, snr_db: float, seed: int, demo: bool, initial: str = "misfire") -> dict:
    """`diagnose` is the app's diagnoser (callable on a Window). Returns plain JSON-able data."""
    index = {name: FEATURE_NAMES.index(name) for name in SHOWN_FEATURES}
    common = dict(rpm=rpm, load=load, snr_db=snr_db)

    healthy = [simulate_window(fault="healthy", severity=0.0, seed=seed + 1000 + k, **common) for k in range(HEALTHY_SEEDS)]
    vectors = np.array([extract_features(w)[0] for w in healthy])
    spread = vectors.std(axis=0)
    # The reference is the healthy window with the same seed as every fault window, so both share the same noise draws
    # and the difference is the fault's effect. The spread of the other healthy windows says what counts as a big move.
    ref_window = simulate_window(fault="healthy", severity=0.0, seed=seed, **common)
    mean = extract_features(ref_window)[0]
    # A feature that barely moves between healthy windows would give a huge ratio; floor its spread at 2% of its size.
    floor = np.maximum(spread, 0.02 * np.abs(mean) + 1e-6)
    sound_scale = float(np.max([np.max(np.abs(w.sound)) for w in healthy]))
    vib_scale = float(np.max([np.max(np.abs(w.vibration)) for w in healthy]))
    rms = {"sound": float(np.sqrt(np.mean(ref_window.sound ** 2))), "vibration": float(np.sqrt(np.mean(ref_window.vibration ** 2)))}

    faults = {}
    for fault in FAULTS:
        sev = 0.0 if fault == "healthy" else float(severity)
        window = simulate_window(fault=fault, severity=sev, seed=seed, **common)
        vector, _ = extract_features(window)
        result = diagnose(window)
        evidence = {}
        for name, i in index.items():
            evidence[name] = 0.0 if fault == "healthy" else float(min(1.0, abs(vector[i] - mean[i]) / floor[i] / 8.0))
        faults[fault] = {
            "label": result.label, "status": result.status, "confidence": float(result.confidence),
            "probs": {k: float(v) for k, v in result.probabilities.items()},
            "severities": {k: float(v) for k, v in result.severities.items()},
            "reasons": list(result.reasons),
            "features": {name: float(vector[i]) for name, i in index.items()},
            "evidence": evidence,
            "rms": {"sound": float(np.sqrt(np.mean(window.sound ** 2))), "vibration": float(np.sqrt(np.mean(window.vibration ** 2)))},
            "wave": {"sound": _envelope(window.sound, sound_scale), "vibration": _envelope(window.vibration, vib_scale)},
        }
    return {
        "meta": {"rpm": float(rpm), "load": float(load), "severity": float(severity), "snr": float(snr_db), "seed": int(seed),
                 "demo": bool(demo), "initial": initial, "faults": list(FAULTS)},
        "healthy": {"mean": {name: float(mean[i]) for name, i in index.items()}, "rms": rms},
        "faults": faults,
    }
