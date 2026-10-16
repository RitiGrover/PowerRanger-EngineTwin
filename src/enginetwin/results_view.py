"""Results-page presentation of existing artifacts, independent of demo mode."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from enginetwin.liu_guard import SCENARIOS, DETECTOR_BY_SENSOR, build_guard_replay, load_guard_bundle


@st.cache_data(show_spinner=False)
def measured_results(path: str, modified_ns: int) -> pd.DataFrame:
    """Summarize the existing guard decisions; do not train or alter the model."""
    bundle = load_guard_bundle(path)
    rows = []
    for scenario_id, scenario in SCENARIOS.items():
        replay = build_guard_replay(bundle, scenario_id)
        detector = DETECTOR_BY_SENSOR[scenario["fault_sensor"]]
        times = replay.frame["window_end_s"]
        before = times < replay.injection_time
        after = ~before
        confirmed = replay.confirmed[detector]
        detections = times[after & confirmed]
        rows.append({
            "Replay": scenario["name"],
            "Sensor bias": scenario["code"],
            "Windows": len(times),
            "Confirmed before injection": int(confirmed[before].sum()),
            "Pre-injection windows": int(before.sum()),
            "Confirmed after injection": int(confirmed[after].sum()),
            "Post-injection windows": int(after.sum()),
            "First post-injection confirmation (s)": (
                float(detections.iloc[0] - replay.injection_time) if len(detections) else None
            ),
        })
    return pd.DataFrame(rows)


def render_available_results(bundle_path: Path, cache: Path) -> None:
    st.caption("Measured replay evidence and saved validation reports, shown separately.")
    if bundle_path.exists():
        try:
            table = measured_results(str(bundle_path), bundle_path.stat().st_mtime_ns)
        except (ValueError, KeyError, OSError) as error:
            st.warning(f"The measured replay report could not be loaded: {error}")
        else:
            st.markdown("### Measured engine replay")
            cols = st.columns(3)
            cols[0].metric("Held-out replays", len(table))
            cols[1].metric("Recorded windows", int(table["Windows"].sum()))
            cols[2].metric("Sensor channels", 10)
            st.dataframe(table, hide_index=True, width="stretch")
            st.caption("Computed from the supplied LiU-ICE guard and held-out recordings. Counts refer to each replay's target sensor. A confirmation before injection is a possible false alarm; a blank delay means no confirmation after injection. These are sensor-bias replay results, not acoustic fault-classification accuracy.")
    else:
        st.info("No measured replay bundle is present in this copy of the project.")

    model_path = bundle_path.with_name("model_bundle.joblib")
    if model_path.exists():
        st.markdown("### Saved model evaluation")
        metrics = saved_model_results(str(model_path), model_path.stat().st_mtime_ns)
        st.dataframe(metrics.style.format({"ROC AUC": "{:.3f}", "F1": "{:.3f}",
                                          "Precision": "{:.1%}", "Recall": "{:.1%}",
                                          "False-alarm rate": "{:.1%}"}), hide_index=True, width="stretch")
        st.caption("Stored held-out evaluation for the supplied active-fault and early-warning classifiers. These classifiers are separate from the sensor-consistency guard above. Scores are read from the bundle; no models are trained or rescored here.")
        with st.expander("How to read these scores"):
            st.write("Recall is the share of faults found; precision is the share of alerts that were correct. False-alarm rate measures alerts on healthy windows. ROC AUC measures ranking ability, not accuracy at the selected alert threshold. Low recall means faults were missed even when precision is high.")

    reports = {
        "Classical fault classifiers": "results_classical.json",
        "OBD network": "results_obd_ann.json",
        "Multimodal network": "results_cnn.json",
        "Health and anomaly detection": "results_health_autoencoder.json",
        "Alert timing": "results_latency.json",
        "Efficiency models": "results_efficiency.json",
    }
    available = [label for label, filename in reports.items() if (cache / filename).exists()]
    st.markdown("### Model validation")
    if available:
        st.caption("Available saved reports are shown below. Acoustic scores use simulated sound and vibration.")
    else:
        st.info("No synthetic-model validation reports are included in this project. Acoustic accuracy and efficiency-validation scores will appear when those reports are supplied.")
    with st.expander("Report availability"):
        st.dataframe(pd.DataFrame([
            {"Report": label, "Status": "Available" if (cache / filename).exists() else "Not included", "File": filename}
            for label, filename in reports.items()
        ]), hide_index=True, width="stretch")


@st.cache_data(show_spinner=False)
def saved_model_results(path: str, modified_ns: int) -> pd.DataFrame:
    bundle = joblib.load(path)
    columns = {"roc_auc": "ROC AUC", "precision": "Precision", "recall": "Recall",
               "f1": "F1", "false_alarm_rate": "False-alarm rate", "test_rows": "Test windows",
               "test_trajectories": "Test trajectories"}
    return pd.DataFrame([
        {"Model": label, **{display: bundle[key]["metrics"].get(field) for field, display in columns.items()}}
        for key, label in (("active", "Active fault"), ("early", "Early warning")) if key in bundle
    ])
