"""EngineTwin dashboard: one app for live diagnosis, signatures, efficiency, health, data and results.

Run with:  streamlit run app.py
Everything it shows is computed from models and tables saved in cache/ (build them with the scripts in scripts/);
nothing is trained while the dashboard runs."""
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from live_integration import render_live
from enginetwin import dashboard as dash
from enginetwin.diagnosis import Diagnoser, DiagnosisResult
from enginetwin.demo import DemoDiagnoser, DemoEfficiencyModel, demo_efficiency_cells, demo_trip_replay
from enginetwin.efficiency_map import efficiency_map
from enginetwin.efficiency_model import KIA_SOUL_MAX_TORQUE_NM
from enginetwin.fault_features import refine_rpm
from enginetwin.health import time_to_service
from enginetwin.latency import simulate_onset
from enginetwin.liu_guard import SCENARIOS as LIU_SCENARIOS, build_guard_replay, load_guard_bundle
from enginetwin.losses import DEFAULT_LOSS_PER_SEVERITY
from enginetwin.ocslab import load_ocslab
from enginetwin.optimiser import optimise
from enginetwin.recording import has_obd, obd_from_csv, window_from_wav
from enginetwin.replay import gear_from_rpm, replay_trip
from enginetwin.results import autoencoder_table, build_results_table, latency_table, load_confusions
from enginetwin.schema import FAULTS
from enginetwin.signals import band_energy_fraction, envelope_spectrum, half_order_ratio, order_spectrum, per_cylinder_energy
from enginetwin.spectrograms import window_spectrogram
from enginetwin.streaming import rolling_windows
from enginetwin.vehicle import VehicleSpec, engine_point, wheel_power_w
from enginetwin.ui import COCKPIT_CSS, engine_panel, event_log, hero, metric_card, scroll_engine_simulation, sensor_grid, showroom_landing, toolbar

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
LIU_GUARD_BUNDLE = ROOT / "data" / "liu_ice" / "guard_bundle.joblib"
SPEC = VehicleSpec()
STREAM_SECONDS, ONSET_S, WINDOW_S = 8.0, 3.0, 2.0
REQUIRED_MODELS = ("model_cnn.pkl", "model_softmax.pkl", "model_autoencoder.pkl", "model_severity.pkl", "baseline.pkl", "model_efficiency_gbm.pkl")
MISSING_MODELS = [name for name in REQUIRED_MODELS if not (CACHE / name).exists()]
DEMO_MODE = bool(MISSING_MODELS)

st.set_page_config(page_title="EngineTwin", layout="wide")
st.markdown(COCKPIT_CSS, unsafe_allow_html=True)

# The product-configurator opening is intentionally isolated from the analytical
# dashboard. This keeps the first impression quiet and avoids loading models
# until the visitor asks to enter the live twin.
if "entered_engine_twin" not in st.session_state:
    st.session_state.entered_engine_twin = False
if st.query_params.get("view") == "cockpit":
    st.session_state.entered_engine_twin = True
if not st.session_state.entered_engine_twin:
    st.markdown(showroom_landing(ROOT / "assets" / "engine-showroom-hero.png"), unsafe_allow_html=True)
    if st.button("Enter live twin", key="enter_engine_twin"):
        st.session_state.entered_engine_twin = True
        st.rerun()
    st.stop()


# ---------------------------------------------------------------- cached resources (loaded once, never trained here)
@st.cache_resource
def diagnoser() -> Diagnoser:
    return DemoDiagnoser() if DEMO_MODE else Diagnoser.load(CACHE)


@st.cache_resource
def efficiency_model():
    return DemoEfficiencyModel() if DEMO_MODE else pickle.load(open(CACHE / "model_efficiency_gbm.pkl", "rb"))


@st.cache_resource
def measured_guard_bundle():
    return load_guard_bundle(LIU_GUARD_BUNDLE)


@st.cache_data
def ocslab_cells() -> pd.DataFrame:
    files = list((ROOT / "data" / "osclab").glob("*.csv"))
    return efficiency_map(load_ocslab(files[0])) if files else demo_efficiency_cells()


@st.cache_data
def make_stream(fault, severity, rpm, load, snr_db, seed):
    return simulate_onset(fault, severity, onset_s=ONSET_S, duration_s=STREAM_SECONDS, rpm=rpm, seed=seed, load=load, snr_db=snr_db)


@st.cache_data
def diagnosis_data_at(fault, severity, rpm, load, snr_db, seed, end_s):
    window = make_stream(fault, severity, rpm, load, snr_db, seed).window(end_s, WINDOW_S)
    # Cached model instances can retain an older result class after a hot reload.
    # Cache its data, not its class identity; all diagnosis values are preserved.
    return asdict(diagnoser().diagnose(window))


def diagnose_at(fault, severity, rpm, load, snr_db, seed, end_s):
    return DiagnosisResult(**diagnosis_data_at(fault, severity, rpm, load, snr_db, seed, end_s))


@st.cache_data
def read_json(name):
    path = CACHE / name
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def trip_replay(name, fault, severity, loss_rates):
    """Healthy replays come from the saved tables; with a fault the first 900 seconds are replayed live with its loss."""
    source = "ocslab" if name == "Ocslab trip H-0 (held-out driver)" else "ved"
    if DEMO_MODE:
        return demo_trip_replay(fault, severity, dict(loss_rates))
    if fault == "healthy":
        return pd.read_csv(CACHE / f"replay_{source}.csv")
    if source == "ocslab":
        trip = load_ocslab(next((ROOT / "data" / "osclab").glob("*.csv")))
        trip = trip[trip.trip_id == "H-0"]
    else:
        from enginetwin.ved import load_ved

        week = sorted((ROOT / "data" / "ved" / "extracted").glob("*.csv"))[0]
        trip = load_ved([week], ROOT / "data" / "ved" / "VED_Static_Data_ICE&HEV.xlsx")
        trip = trip[trip.trip_id == read_json("replay_summary.json")["ved"]["trip_id"]].sort_values("time_s")
    table, _ = replay_trip(trip.head(900), efficiency_model(), SPEC, source, fault, severity, dict(loss_rates))
    return table


def summarise(table):
    litres = lambda col: float((table[col] * table["time_s"].diff().fillna(1.0)).sum() / 3600)
    actual, optimised = litres("actual_fuel_l_h"), litres("opt_fuel_l_h")
    return actual, optimised


SECTIONS = ["Live engine", "Sensor fusion", "Engine insights", "Real data lab", "Validation"]
# Keep page-specific inputs when their widgets are temporarily off screen.
for input_key in list(st.session_state):
    if input_key in {"guard_enabled", "triggered_failure", "health_day", "heat_fault", "confusion", "liu_scenario", "liu_cursor"} or input_key.startswith("manual_bias_"):
        st.session_state[input_key] = st.session_state[input_key]

st.markdown("""
<style>
[data-testid="stSidebar"] { min-width:290px; max-width:290px; }
[data-testid="stSidebarContent"] { padding:1.15rem 1rem .8rem; }
[data-testid="stSidebar"] h1 { font-size:1.42rem; margin:0 0 .1rem; }
[data-testid="stSidebar"] [role="radiogroup"] { gap:.18rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding:.46rem .55rem; border-radius:7px; border:1px solid transparent;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background:#24251f; border-color:#59513a;
}
.et-side-kicker { color:#d8b453; font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; margin-top:.85rem; }
.et-side-note { color:#9e9e96; font-size:.76rem; line-height:1.35; margin:.18rem 0 .5rem; }
.et-demo { max-width:100%; overflow:visible!important; white-space:normal!important; }
.et-demo span { display:block; overflow:visible!important; white-space:normal!important; }
@media(max-height:760px) {
  [data-testid="stSidebarContent"] { padding-top:.65rem; }
  [data-testid="stSidebar"] [role="radiogroup"] label { padding:.3rem .45rem; }
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("EngineTwin")
    st.caption("Multisensor fault intelligence")
    st.markdown('<div class="et-side-kicker">Workspace</div>', unsafe_allow_html=True)
    workspace = st.radio("Workspace", SECTIONS, key="workspace_section", label_visibility="collapsed")
    section = workspace
    st.divider()

# ---------------------------------------------------------------- sidebar: controls
if workspace == "Live engine":
    with st.sidebar:
        st.markdown('<div class="et-side-kicker">Fault injection</div>', unsafe_allow_html=True)
        st.markdown('<div class="et-side-note">Choose one measured benchmark fault and inject it into the running replay.</div>', unsafe_allow_html=True)
        live_fault = st.selectbox(
            "Fault type",
            list(LIU_SCENARIOS),
            format_func=lambda key: f"{LIU_SCENARIOS[key]['name']} · {LIU_SCENARIOS[key]['code']}",
            key="sidebar_live_fault",
        )
        inject_col, clear_col = st.columns(2)
        if inject_col.button("Inject", key="sidebar_inject", type="primary", use_container_width=True):
            st.session_state.guard_model_version = "manual-all-sensors-v1"
            st.session_state.scenario_id = live_fault
            st.session_state.triggered_failure = True
            st.session_state.triggered_event = None
            st.session_state.manual_event = None
            st.session_state.guard_tripped = False
            st.session_state.engine_failed = False
            st.session_state.sim_running = True
        if clear_col.button("Clear", key="sidebar_clear", use_container_width=True):
            st.session_state.triggered_failure = False
            st.session_state.triggered_event = None
            st.session_state.manual_event = None
            st.session_state.guard_tripped = False
            st.session_state.engine_failed = False

if workspace == "Engine insights":
    with st.sidebar:
        st.markdown('<div class="et-side-kicker">Analysis</div>', unsafe_allow_html=True)
        section = st.radio("Analysis", ["Sweet spot", "Health"], key="insight_section", label_visibility="collapsed")
        if section == "Sweet spot":
            drive_mode = st.selectbox("Driving condition", list(dash.DRIVE_MODES), index=1, key="drive_mode")

if workspace == "Validation":
    with st.sidebar:
        st.markdown('<div class="et-side-kicker">Evidence</div>', unsafe_allow_html=True)
        section = st.radio("Evidence", ["Results", "Data explorer"], key="validation_section", label_visibility="collapsed")

if workspace == "Real data lab":
    section = "Real recording"

if workspace == "Sensor fusion":
    with st.sidebar:
        st.markdown('<div class="et-side-kicker">Simulation</div>', unsafe_allow_html=True)
        st.markdown('<div class="et-side-note">A controlled fault lets you see every signal change before the diagnosis changes.</div>', unsafe_allow_html=True)
        injected = st.selectbox("Inject a fault", FAULTS, key="fault", format_func=lambda value: value.replace("+", " + ").replace("_", " ").title())
        severity = st.slider("Fault severity", 0.0, 1.0, 0.8, 0.05, key="severity", disabled=injected == "healthy")
        snr_db = int(st.session_state.get("snr", 20))
else:
    injected = st.session_state.get("fault", "healthy")
    severity = float(st.session_state.get("severity", 0.8))
    snr_db = int(st.session_state.get("snr", 20))

drive_mode = st.session_state.get("drive_mode", list(dash.DRIVE_MODES)[1])
seed = int(st.session_state.get("seed", 7))
loss_rates = {
    "misfire": float(st.session_state.get("loss_misfire", int(DEFAULT_LOSS_PER_SEVERITY["misfire"] * 100))) / 100,
    "knock": float(st.session_state.get("loss_knock", int(DEFAULT_LOSS_PER_SEVERITY["knock"] * 100))) / 100,
    "valve_wear": float(st.session_state.get("loss_valve", int(DEFAULT_LOSS_PER_SEVERITY["valve_wear"] * 100))) / 100,
}
trip_name = st.session_state.get("trip", "Synthetic urban trip" if DEMO_MODE else "Ocslab trip H-0 (held-out driver)")

mode = dash.drive_mode_point(drive_mode)
sev = 0.0 if injected == "healthy" else float(severity)
params = (injected, sev, mode["rpm"], mode["load"], float(snr_db), int(seed))
stream = make_stream(*params)

st.markdown(hero(), unsafe_allow_html=True)
if DEMO_MODE and workspace in {"Sensor fusion", "Engine insights"}:
    st.markdown(
        '<div class="et-demo"><b>SIMULATION LAB</b><span>Sound and vibration are controlled signals for explaining the pipeline. Measured validation is kept separately in Real data lab and Validation.</span></div>',
        unsafe_allow_html=True,
    )
steps = [round(WINDOW_S + 0.5 * k, 1) for k in range(int((STREAM_SECONDS - WINDOW_S) / 0.5) + 1)]
if workspace == "Sensor fusion":
    end_s = st.select_slider("Playback position", options=steps, value=5.0, key="position", format_func=lambda value: f"{value:.1f} s")
else:
    end_s = float(st.session_state.get("position", 5.0))
result = diagnose_at(*params, end_s)
window = stream.window(end_s, WINDOW_S)
rpm_now = refine_rpm(window.sound, window.sound_fs, window.obd["engine_speed"])


def status_banner(res):
    if res.status == "Healthy":
        st.success(f"Healthy (confidence {res.confidence:.0%})")
    else:
        st.error(f"Fault: {res.label.replace('+', ' and ').replace('_', ' ')} (confidence {res.confidence:.0%})")


# ---------------------------------------------------------------- Live engine
if section == "Live engine":
    render_live()

if section == "Synthetic engine":
    st.markdown(toolbar(drive_mode, end_s, result.label, int(snr_db)), unsafe_allow_html=True)
    st.markdown(scroll_engine_simulation(result.label, rpm_now, window.obd["calculated_load_value"]), unsafe_allow_html=True)
    trim = window.obd["short_term_fuel_trim_bank1"] + window.obd["long_term_fuel_trim_bank1"]
    kpis = st.columns(4)
    kpi_data = (
        ("Engine speed", f"{rpm_now:,.0f}", "RPM", "Crankshaft angular velocity", "#d8aa32"),
        ("Calculated load", f"{window.obd['calculated_load_value']:.1f}", "%", "Combustion demand", "#b78d2c"),
        ("Coolant temperature", f"{window.obd['engine_coolant_temperature']:.1f}", "°C", "Thermal operating state", "#f6b93b"),
        ("AI guard state", result.status.upper(), "", f"{result.confidence:.0%} confidence", "#e6bc55" if result.status == "Healthy" else "#d78b80"),
    )
    for column, data in zip(kpis, kpi_data):
        column.markdown(metric_card(*data), unsafe_allow_html=True)

    stop = int(round(end_s * stream.fs))
    start = max(0, stop - int(2.5 * stream.fs))
    stride = max(1, (stop - start) // 360)
    chart_t = np.arange(start, stop, stride) / stream.fs
    sound_trace = stream.sound[start:stop:stride]
    vibration_trace = stream.vibration[start:stop:stride]
    norm = lambda values, offset: offset + 12 * values / (np.std(values) + 1e-9)

    trace_fig = go.Figure()
    trace_fig.add_trace(go.Scatter(x=chart_t, y=norm(sound_trace, 68), name="Acoustic", line={"color": "#d8b453", "width": 1.6}))
    trace_fig.add_trace(go.Scatter(x=chart_t, y=norm(vibration_trace, 38), name="Vibration", line={"color": "#9ba9ba", "width": 1.5}))
    trace_fig.add_trace(go.Scatter(x=chart_t, y=np.full_like(chart_t, window.obd["calculated_load_value"]), name="Engine load", line={"color": "#a1b3a2", "width": 1.4}))
    if result.status != "Healthy":
        trace_fig.add_trace(go.Scatter(x=chart_t, y=np.full_like(chart_t, 8), name="Fault evidence", line={"color": "#d78b80", "width": 2, "dash": "dash"}))
    trace_fig.update_layout(
        title={"text": "SYNCHRONIZED SENSOR STREAMS & FAULT EVIDENCE", "font": {"size": 13, "family": "Arial"}},
        paper_bgcolor="rgba(7,23,34,.75)", plot_bgcolor="#151614", font={"color": "#b3b3aa", "family": "Arial"},
        height=388, margin={"l": 35, "r": 18, "t": 54, "b": 36}, legend={"orientation": "h", "y": 1.12, "x": .46},
        xaxis={"title": "Replay time · seconds", "gridcolor": "#30312d"}, yaxis={"title": "normalized signal", "gridcolor": "#30312d", "range": [0, 100]},
    )
    left, right = st.columns([1.72, 1.0], gap="small")
    with left:
        st.plotly_chart(trace_fig, width="stretch", config={"displayModeBar": False})
        readings = (
            ("Engine speed", f"{rpm_now:,.0f}", "RPM", "STREAMING", "#d8aa32"),
            ("Air mass flow", f"{window.obd['maf']:.2f}", "g/s", "NOMINAL", "#d8aa32"),
            ("Timing advance", f"{window.obd['timing_advance']:.1f}", "deg", "STREAMING", "#d8aa32"),
            ("Fuel trim", f"{trim:+.1f}", "%", "WATCH" if abs(trim) > 8 else "NOMINAL", "#f6b93b"),
            ("Noise floor", f"{snr_db}", "dB", "SYNCHRONIZED", "#d8aa32"),
            ("Throttle proxy", f"{window.obd['calculated_load_value']:.1f}", "%", "STREAMING", "#d8aa32"),
            ("Coolant", f"{window.obd['engine_coolant_temperature']:.1f}", "°C", "NOMINAL", "#d8aa32"),
            ("Fault evidence", f"{max(result.probabilities.values()):.0%}", "", result.label.upper(), "#d78b80" if result.status != "Healthy" else "#d8aa32"),
            ("Engine load", f"{window.obd['calculated_load_value']:.1f}", "%", "STREAMING", "#d8aa32"),
            ("Playback", f"{end_s:.1f}", "s", "ACTIVE", "#d8aa32"),
        )
        st.markdown(sensor_grid(readings), unsafe_allow_html=True)
    with right:
        st.markdown(engine_panel(result.label, float(np.mean(list(result.severities.values()))) if result.severities else 0.0,
                                 rpm_now, window.obd["calculated_load_value"]), unsafe_allow_html=True)
        st.markdown(event_log(result.label, result.confidence, end_s), unsafe_allow_html=True)
    if not has_obd(window):
        st.warning("No OBD readings.")

# ---------------------------------------------------------------- Sensor fusion story
if section == "Sensor fusion":
    st.subheader("How the sensors become a diagnosis")
    st.caption("The selected fault enters at 3.0 s. Move the playback control to watch raw sound, vibration and ECU telemetry turn into signatures, evidence and a decision.")

    healthy_window = make_stream("healthy", 0.0, mode["rpm"], mode["load"], float(snr_db), int(seed)).window(end_s, WINDOW_S)
    trim_now = window.obd["short_term_fuel_trim_bank1"] + window.obd["long_term_fuel_trim_bank1"]
    trim_base = healthy_window.obd["short_term_fuel_trim_bank1"] + healthy_window.obd["long_term_fuel_trim_bank1"]
    fusion_metrics = [
        ("Acoustic energy", float(np.sqrt(np.mean(window.sound ** 2))), float(np.sqrt(np.mean(healthy_window.sound ** 2))), "microphone"),
        ("Vibration energy", float(np.sqrt(np.mean(window.vibration ** 2))), float(np.sqrt(np.mean(healthy_window.vibration ** 2))), "accelerometer"),
        ("Half-order rhythm", float(half_order_ratio(window.sound, window.sound_fs, rpm_now)), float(half_order_ratio(healthy_window.sound, healthy_window.sound_fs, mode["rpm"])), "misfire signature"),
        ("Knock-band energy", float(band_energy_fraction(window.sound, window.sound_fs, 5500, 7500)), float(band_energy_fraction(healthy_window.sound, healthy_window.sound_fs, 5500, 7500)), "combustion signature"),
        ("Valve-band energy", float(band_energy_fraction(window.vibration, window.vibration_fs, 2500, 3500)), float(band_energy_fraction(healthy_window.vibration, healthy_window.vibration_fs, 2500, 3500)), "valvetrain signature"),
        ("Fuel-trim shift", float(trim_now), float(trim_base), "ECU response"),
    ]
    deltas = [float(np.clip((current - baseline) / (abs(baseline) + 1e-6), -4, 4)) for _, current, baseline, _ in fusion_metrics]
    strongest = sorted(zip(fusion_metrics, deltas), key=lambda item: abs(item[1]), reverse=True)[:3]

    flow = go.Figure(go.Sankey(
        arrangement="fixed",
        node={
            "label": ["Microphone", "Accelerometer", "ECU / OBD", "Frequency signatures", "Combustion rhythm", "Telemetry shifts", "Multimodal model", result.label.replace("+", " + ").replace("_", " ").title()],
            "x": [0.02, 0.02, 0.02, 0.34, 0.34, 0.34, 0.68, 0.96],
            "y": [0.08, 0.45, 0.82, 0.14, 0.5, 0.82, 0.48, 0.48],
            "color": ["#d8b453", "#9ba9ba", "#a1b3a2", "#52472d", "#52472d", "#3c4941", "#d8b453", "#d78b80" if result.status != "Healthy" else "#a1b3a2"],
            "pad": 18,
            "thickness": 18,
        },
        link={
            "source": [0, 0, 1, 1, 2, 3, 4, 5, 6],
            "target": [3, 4, 3, 4, 5, 6, 6, 6, 7],
            "value": [2.1, 1.5, 1.8, 1.4, 2.2, 2.0, 1.8, 2.2, 5.8],
            "color": ["rgba(216,180,83,.24)", "rgba(216,180,83,.18)", "rgba(155,169,186,.22)", "rgba(155,169,186,.18)", "rgba(161,179,162,.22)", "rgba(216,180,83,.20)", "rgba(216,180,83,.20)", "rgba(161,179,162,.20)", "rgba(215,139,128,.30)" if result.status != "Healthy" else "rgba(161,179,162,.30)"],
        },
    ))
    flow.update_layout(
        height=330, margin={"l": 8, "r": 8, "t": 12, "b": 8},
        paper_bgcolor="#151614", plot_bgcolor="#151614", font={"color": "#f5f4ef", "size": 12},
    )
    st.plotly_chart(flow, width="stretch", config={"displayModeBar": False})

    stage_cols = st.columns([1.1, 1.1, 1.1, 1.25])
    source_cards = (
        ("01 · SENSE", "3 modalities", "sound · vibration · ECU"),
        ("02 · EXTRACT", "6 live signatures", strongest[0][0][0]),
        ("03 · FUSE", f"{max(result.probabilities.values()):.0%} evidence", "CNN + explainable features"),
        ("04 · DECIDE", result.status.upper(), result.label.replace("+", " + ").replace("_", " ")),
    )
    for column, (label, value, note) in zip(stage_cols, source_cards):
        column.markdown(metric_card(label, value, "", note, "#d78b80" if result.status != "Healthy" and label.startswith("04") else "#d8b453"), unsafe_allow_html=True)

    left, right = st.columns([1.35, 1], gap="medium")
    with left:
        comparison = go.Figure()
        labels = [item[0] for item in fusion_metrics]
        comparison.add_trace(go.Bar(y=labels, x=deltas, orientation="h", marker_color=["#d78b80" if value > .2 else "#9ba9ba" if value < -.2 else "#a1b3a2" for value in deltas], hovertemplate="%{y}<br>%{x:+.1f}× vs healthy<extra></extra>"))
        comparison.add_vline(x=0, line_color="#77776f")
        comparison.update_layout(title="CHANGE FROM THE MATCHED HEALTHY BASELINE", height=355, margin={"l": 15, "r": 15, "t": 48, "b": 35}, paper_bgcolor="#151614", plot_bgcolor="#151614", font={"color": "#b4b3aa"}, xaxis={"title": "relative change · capped at ±4×", "gridcolor": "#30312d"}, yaxis={"autorange": "reversed", "gridcolor": "#30312d"}, showlegend=False)
        st.plotly_chart(comparison, width="stretch", config={"displayModeBar": False})
    with right:
        status_banner(result)
        st.markdown("**What moved the decision**")
        for (name, _, _, meaning), delta in strongest:
            direction = "rose" if delta >= 0 else "fell"
            st.markdown(f"- **{name}** {direction} by {abs(delta):.1f}× relative to the matched healthy window ({meaning}).")
        for sentence in result.reasons[:2]:
            st.markdown(f"- {sentence}")
        st.plotly_chart(dash.probability_figure(result.probabilities, "simulated"), width="stretch", config={"displayModeBar": False})

    with st.expander("Inspect the frequency-domain evidence"):
        spec = window_spectrogram(window)
        spectra = st.columns(2)
        spectra[0].plotly_chart(dash.spectrogram_figure(spec[0], WINDOW_S, "Sound", "simulated"), width="stretch")
        spectra[1].plotly_chart(dash.spectrogram_figure(spec[1], WINDOW_S, "Vibration", "simulated"), width="stretch")

# ---------------------------------------------------------------- Diagnosis
if section == "Diagnosis":
    st.subheader("Diagnosis")
    status_banner(result)
    left, right = st.columns(2)
    left.plotly_chart(dash.probability_figure(result.probabilities, "simulated"), width="stretch")
    with right:
        st.markdown("**Why**")
        for sentence in result.reasons:
            st.markdown(f"- {sentence}")
        for fault, value in result.severities.items():
            st.metric(f"Estimated severity of {fault.replace('_', ' ')}", f"{value:.0%}")
        if DEMO_MODE:
            st.caption(f"Demo baseline says: {result.softmax_label}. Pattern score {result.anomaly_score:.2f}; demo alarm level {result.anomaly_threshold:.2f}.")
        else:
            st.caption(f"Explainable baseline (softmax on hand-made features) says: {result.softmax_label}. "
                       f"Autoencoder: {'unfamiliar' if result.anomaly else 'familiar'} (score {result.anomaly_score:.2f}, alarm level {result.anomaly_threshold:.2f}).")
    detected = [f for f in result.probabilities if f in result.label]
    if detected:
        fault = st.selectbox("Show what triggered", detected, key="heat_fault")
        from enginetwin.spectrograms import standardise_spectrogram

        heat = diagnoser().heatmap(window, fault)
        spec = window_spectrogram(window)
        st.plotly_chart(dash.heat_overlay_figure(standardise_spectrogram(spec)[0], heat, "simulated"), width="stretch")
        st.caption("Demo highlight for interface testing." if DEMO_MODE else
                   "Heat map: how much the fault probability drops when each patch of the spectrogram is hidden (sound and vibration together).")
    else:
        st.info("No fault detected, so there is no trigger to show.")

# ---------------------------------------------------------------- Sweet spot
if section == "Sweet spot":
    st.subheader("Sweet spot")
    cells = ocslab_cells()
    torque = mode["load"] * KIA_SOUL_MAX_TORQUE_NM
    wheel_power = wheel_power_w(mode["speed_kmh"], 0.0, SPEC)
    rpm_c, torque_c = engine_point(mode["speed_kmh"], gear_from_rpm(mode["rpm"], mode["speed_kmh"], SPEC), wheel_power, SPEC)
    fault_label = result.label
    fault_severity = float(np.mean(list(result.severities.values()))) if result.severities else 0.0
    rec = optimise(mode["speed_kmh"], wheel_power, efficiency_model(), SPEC, fault_label, fault_severity, loss_rates,
                   actual_gear=gear_from_rpm(mode["rpm"], mode["speed_kmh"], SPEC))
    cols = st.columns(4)
    cols[0].metric("Recommended gear", f"{rec['gear']}", f"now {rec['actual']['gear']}")
    cols[1].metric("Recommended engine speed", f"{rec['rpm']:.0f} rpm", f"{rec['rpm'] - rec['actual']['rpm']:+.0f} rpm")
    cols[2].metric("Efficiency gain", f"{rec['efficiency_gain'] * 100:+.1f} points")
    cols[3].metric("Fuel saved", f"{rec['fuel_saved_l_per_h']:.2f} L/h", f"Rs {rec['rupees_saved_per_h']:.0f} per hour")
    if fault_label != "healthy":
        st.warning(f"With the diagnosed {fault_label.replace('+', ' and ').replace('_', ' ')} (severity {fault_severity:.0%}) the engine loses "
                   f"{rec['fault_loss_fraction']:.0%} of its efficiency ({rec['efficiency_lost_to_fault'] * 100:.1f} points), burning "
                   f"{rec['fault_extra_fuel_l_per_h']:.2f} L/h more than a healthy one. These loss rates are assumptions you can change in the sidebar.")
    map_source = "simulated" if DEMO_MODE else "real"
    st.plotly_chart(dash.efficiency_map_figure(cells, (rec["actual"]["rpm"], rec["actual"]["torque_nm"]), (rec["rpm"], rec["torque_nm"]), map_source), width="stretch")
    if DEMO_MODE:
        st.caption("Demo efficiency surface: the current point is white and the recommended point is gold. Replace demo mode with trained artifacts before reporting these values.")
    else:
        st.caption("The map is measured Ocslab data (Kia Soul, 10 drivers). The points are where the engine runs now (white) and where the optimiser would run it (gold) "
                   "for the same power, using an efficiency model learned from Ocslab. The Ocslab fuel unit is inferred as microlitres per second, "
                   "and absolute efficiencies are about half of EPA's. Read the shape and gains rather than the absolute level.")
    st.markdown("**Scenario replay: actual against optimised**" if DEMO_MODE else "**Replay of a real trip: actual against optimised**")
    table = trip_replay(trip_name, injected if injected != "healthy" else "healthy", sev, tuple(loss_rates.items()))
    actual_l, optimised_l = summarise(table)
    cols = st.columns(3)
    cols[0].metric("Fuel actually used", f"{actual_l:.3f} L")
    cols[1].metric("Fuel with optimised gears", f"{optimised_l:.3f} L")
    cols[2].metric("Saved", f"{actual_l - optimised_l:.3f} L", f"Rs {(actual_l - optimised_l) * 105:.1f}")
    st.plotly_chart(dash.series_figure(table, "mixed"), width="stretch")
    if DEMO_MODE:
        st.caption("Simulated scenario: both series are generated for interface testing. The values are illustrative, not measured savings.")
    else:
        st.caption("Mixed: speed, rpm and fuel are recorded values; the optimised series is modelled. "
                   + ("The VED trip has no torque. Its power comes from a vehicle model and the efficiency model did not validate on VED. Treat that saving as an illustration."
                      if trip_name.startswith("VED") else "The saving on this held-out Ocslab trip is small: the car's automatic already picks gears close to what the model prefers."))

# ---------------------------------------------------------------- Health
if section == "Health":
    st.subheader("Health")
    health = read_json("results_health_autoencoder.json")
    if health is None and DEMO_MODE:
        demo_days = np.arange(61, dtype=float)
        demo_true = 100.0 * (1.0 - (demo_days / 60.0) ** 1.6)
        demo_estimated = np.clip(demo_true + 2.0 * np.sin(demo_days / 5.0), 0.0, 100.0)
        health = {
            "wear_series": {"days": demo_days.tolist(), "true_health": demo_true.tolist(), "estimated_health": demo_estimated.tolist()},
            "wear_regressor_mae_test_same": float(np.mean(np.abs(demo_true - demo_estimated)) / 100.0),
        }
    if health:
        series = health["wear_series"]
        days = np.array(series["days"])
        today = st.slider("Day", 5, int(days[-1]), 40, key="health_day")
        estimated = np.array(series["estimated_health"])
        ahead = time_to_service(days[: today + 1], estimated[: today + 1], threshold=20, last=14)
        service_day = today + ahead
        cols = st.columns(3)
        cols[0].metric("Health index today", f"{estimated[today]:.0f} / 100")
        cols[1].metric("Predicted service day", f"{service_day:.0f}" if np.isfinite(service_day) else "none")
        cols[2].metric("Days left (estimate)", f"{ahead:.0f}" if np.isfinite(ahead) else "no wear trend")
        st.plotly_chart(dash.health_figure(days, series["true_health"], estimated, service_day, "simulated"), width="stretch")
        if DEMO_MODE:
            st.caption("Demo wear curve for interface testing. It is synthetic and is not a trained service-life prediction.")
        else:
            st.caption("A simulated valve wears from new to fully worn over 60 days (the wear curve is an assumption). Health is estimated from the vibration "
                       f"alone (error {health['wear_regressor_mae_test_same']:.3f} in severity). The forecast extends the last 14 days in a straight line. "
                       "it is early or late while wear accelerates: it only becomes accurate in the last third.")
    else:
        st.info("Run scripts/train_health_and_autoencoder.py to build the health series.")

# ---------------------------------------------------------------- Data explorer
if section == "Data explorer":
    st.subheader("Data explorer")
    st.dataframe(dash.data_provenance(), hide_index=True, width="stretch")
    st.markdown("**Training and demonstration data are different.** The efficiency model learns from 8 Ocslab drivers and is shown on the other 2 and on VED cars. "
                "The fault models learn from simulated training windows and are shown on freshly simulated streams with other seeds and other noise.")
    st.markdown("**What is real and what is simulated.** The efficiency map, trip features and OBD calibration are real. Sound and vibration are simulated from assumed "
                "fault signatures, so scores on simulated test sets show that the models find the patterns we put in, not accuracy on a real engine.")

# ---------------------------------------------------------------- Real recording
if section == "Real recording":
    st.subheader("Real data lab")
    st.caption("A working replay from a public combustion-engine test bench, followed by an optional recording inspector.")
    if LIU_GUARD_BUNDLE.exists():
        dataset_left, dataset_right = st.columns([1.35, 1])
        with dataset_left:
            real_scenario = st.selectbox(
                "Measured fault run",
                list(LIU_SCENARIOS),
                format_func=lambda key: f"{LIU_SCENARIOS[key]['name']} · {LIU_SCENARIOS[key]['code']}",
                key="real_scenario",
            )
        replay = build_guard_replay(measured_guard_bundle(), real_scenario)
        with dataset_right:
            real_cursor = st.slider("Replay position", 0, len(replay.frame) - 1, min(30, len(replay.frame) - 1), key="real_cursor")
        row = replay.frame.iloc[real_cursor]
        detector = {"Intercooler_pressure": "f_pic", "intake_manifold_pressure": "f_pim", "air_mass_flow": "f_waf"}[replay.scenario["fault_sensor"]]
        confirmed = bool(replay.confirmed.iloc[real_cursor][detector])
        evidence_now = float(replay.evidence.iloc[real_cursor][detector])
        health_now = float(replay.health.iloc[real_cursor])
        elapsed = float(row["window_end_s"])
        measured_rpm = float(row["engine_speed__mean"] * 60 / (2 * np.pi))
        measured_cards = st.columns(4)
        for column, card in zip(measured_cards, (
            ("Bench time", f"{elapsed:.0f}", "s", "five-second measured windows", "#d8aa32"),
            ("Engine speed", f"{measured_rpm:,.0f}", "RPM", "measured crank speed", "#d8aa32"),
            ("Fault evidence", f"{evidence_now:.2f}", "×", "confirmation limit 1.25×", "#d78b80" if confirmed else "#d8aa32"),
            ("Decision", "FAULT" if confirmed else "MONITOR", "", replay.scenario["code"], "#d78b80" if confirmed else "#a1b3a2"),
        )):
            column.markdown(metric_card(*card), unsafe_allow_html=True)

        pre_fault = replay.frame.loc[replay.frame["window_end_s"] < replay.injection_time]
        channel_map = {
            "Manifold pressure": "intake_manifold_pressure__mean",
            "Intercooler pressure": "Intercooler_pressure__mean",
            "Air mass flow": "air_mass_flow__mean",
            "Engine speed": "engine_speed__mean",
            "Fuel mass": "injected_fuel_mass__mean",
            "Throttle": "throttle_position__mean",
        }
        convergence = go.Figure()
        colours = ["#d8b453", "#9ba9ba", "#a1b3a2", "#c7b796", "#d78b80", "#7f8c8d"]
        shown = replay.frame.iloc[: real_cursor + 1]
        for (label, column), colour in zip(channel_map.items(), colours):
            baseline = float(pre_fault[column].median())
            relative = 100 * (shown[column] - baseline) / (abs(baseline) + 1e-9)
            convergence.add_trace(go.Scatter(x=shown["window_end_s"], y=relative, name=label, line={"width": 2, "color": colour}))
        convergence.add_vline(x=replay.injection_time, line_color="#d78b80", line_dash="dash", annotation_text="fault begins")
        convergence.update_layout(
            title="HOW ONE FAULT PROPAGATES ACROSS SIX MEASURED CHANNELS",
            height=440, margin={"l": 35, "r": 18, "t": 58, "b": 105},
            paper_bgcolor="#151614", plot_bgcolor="#151614", font={"color": "#b4b3aa"},
            legend={"orientation": "h", "y": -.34, "x": 0}, xaxis={"title": "test-bench time · seconds", "gridcolor": "#30312d"},
            yaxis={"title": "change from pre-fault median · %", "gridcolor": "#30312d"},
        )
        chart_col, decision_col = st.columns([1.65, 1], gap="medium")
        chart_col.plotly_chart(convergence, width="stretch", config={"displayModeBar": False})
        with decision_col:
            st.markdown("**Measured → expected → residual → decision**")
            st.markdown(f"The healthy consistency model predicts **{replay.scenario['name'].lower()}** from the other channels. Its residual is now **{evidence_now:.2f}×** the learned healthy limit.")
            st.progress(min(evidence_now / 1.25, 1.0), text="Evidence toward confirmation")
            st.metric("Residual-derived health", f"{health_now:.0f} / 100")
            if confirmed:
                st.error("Fault confirmed: two of the last three windows crossed the limit and the current window exceeded 1.25×.")
            elif elapsed < replay.injection_time:
                st.success("Healthy baseline: all channels remain mutually consistent.")
            else:
                st.warning("Fault has started; the guard is collecting consecutive evidence before confirmation.")
        st.markdown(
            "Source: [LiU-ICE Industrial Fault Diagnosis Benchmark](https://vehsys.gitlab-pages.liu.se/dx_safeprocess_benchmarks/liuice/), real internal-combustion-engine test-bench data with nominal and faulty operation."
        )
    else:
        st.info("The prepared LiU‑ICE benchmark bundle is not available in this installation.")

    st.divider()
    st.markdown("### Inspect your own engine recording")
    st.caption("Upload a WAV to inspect its spectrum and run the experimental acoustic pipeline. This is not a field diagnosis: the current acoustic fault labels were learned from controlled synthetic signatures.")
    def remember_upload(key):
        st.session_state[f"saved_{key}"] = st.session_state[key]

    wav = st.file_uploader("Engine recording (WAV)", type=["wav"], key="wav", on_change=remember_upload, args=("wav",))
    obd_file = st.file_uploader("OBD log (CSV, optional)", type=["csv"], key="obd_csv", on_change=remember_upload, args=("obd_csv",))
    wav = wav or st.session_state.get("saved_wav")
    obd_file = obd_file or st.session_state.get("saved_obd_csv")
    if wav is not None:
        st.caption(f"Selected recording: {wav.name}")
        if st.button("Clear recording and OBD log"):
            for upload_key in ("saved_wav", "saved_obd_csv", "wav", "obd_csv"):
                st.session_state.pop(upload_key, None)
            st.rerun()
    if wav is not None:
        try:
            obd = obd_from_csv(obd_file) if obd_file is not None else None
            uploaded = window_from_wav(wav.getvalue(), obd=obd)
            res = diagnoser().diagnose(uploaded)
            status_banner(res)
            if not res.obd_available:
                st.info(f"No OBD log: engine speed was estimated from the sound ({uploaded.obd['engine_speed']:.0f} rpm) and OBD readings are neutral. "
                        "Strong misfire can make the estimate read at half the true speed.")
            for sentence in res.reasons:
                st.markdown(f"- {sentence}")
            st.plotly_chart(dash.probability_figure(res.probabilities, "real"), width="stretch")
            st.plotly_chart(dash.spectrogram_figure(window_spectrogram(uploaded)[0], 2.0, "Recording", "real"), width="stretch")
        except ValueError as error:
            st.error(str(error))

# ---------------------------------------------------------------- Results
if section == "Results":
    st.subheader("Results")
    from enginetwin.results_view import render_available_results
    render_available_results(LIU_GUARD_BUNDLE, CACHE)
    table = build_results_table(CACHE)
    if len(table):
        pivot = table.pivot(index="model", columns="split", values="accuracy")
        order = [c for c in ("train", "validation", "test_same", "test_noise_15db", "test_unseen_rpm", "test_heavy_noise", "test_noise_0db") if c in pivot]
        st.markdown("**Accuracy (7 fault labels)**")
        st.dataframe(pivot[order].style.format("{:.1%}"), width="stretch")
        st.caption("test_noise_15db is inside the trained noise range (10 to 25 dB); test_heavy_noise is 5 dB and test_noise_0db is 0 dB, both beyond it. "
                   "test_unseen_rpm uses 3000 to 5000 rpm, never seen in training. The OBD-only network scores low on purpose: OBD alone cannot see valve wear and barely sees knock.")
        confusions = load_confusions(CACHE)
        choice = st.selectbox("Confusion matrix", sorted(confusions), format_func=lambda k: f"{k[0]} on {k[1]}", key="confusion")
        st.plotly_chart(dash.confusion_figure(confusions[choice], list(FAULTS), f"{choice[0]} on {choice[1]}", "simulated"), width="stretch")
    ae = autoencoder_table(CACHE)
    if len(ae):
        st.markdown("**Healthy-baseline autoencoder** (noise-aware alarm level)")
        st.dataframe(ae.pivot(index="fault", columns="split", values="flagged").style.format("{:.0%}"), width="stretch")
        st.dataframe(ae.drop_duplicates("split").set_index("split")[["false_alarm"]].style.format("{:.0%}"), width="stretch")
        st.caption("Share of faulty windows flagged, and healthy windows wrongly flagged. It is reliable down to roughly 7 to 10 dB and fails at 5 dB and 0 dB.")
    lat = latency_table(CACHE)
    if len(lat):
        st.markdown("**Alert latency** (fault starts at 3 s; 0.5 s chunks, 2 s window, two alerts in a row to confirm)")
        st.dataframe(lat, hide_index=True, width="stretch")
        st.caption("Target under 1 s is not met: the floor is 1 s with two-chunk confirmation. Latencies of models with a high false-alarm rate are not meaningful.")
    eff = read_json("results_efficiency.json")
    if eff:
        st.markdown("**Efficiency model**")
        rows = [{"model": m, "held-out Ocslab drivers MAE": eff[m]["ocslab_held_out"]["mae"], "R2": eff[m]["ocslab_held_out"]["r2"],
                 "VED MAE": eff[m]["ved"]["mae"], "VED R2": eff[m]["ved"]["r2"]} for m in ("gbm", "ann")]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Held-out drivers: the error is about 7% of the mean efficiency (0.15). On VED the model does not validate: VED has no torque column, "
                   "so its efficiency comes from an assumed torque (load times 157 Nm) and MAF-based fuel.")
        st.markdown("Feature importance (increase in error when shuffled): " + ", ".join(f"{k} {v:.4f}" for k, v in eff["gbm"]["importance_ocslab_held_out"].items()))

# ---------------------------------------------------------------- Measured LiU-ICE guard
if section == "Measured guard":
    st.subheader("Measured sensor-consistency guard")
    st.caption("Held-out LiU-ICE engine-bench replay. The ten telemetry channels and injected sensor bias are measured; the health index is derived from model residuals.")
    if not LIU_GUARD_BUNDLE.exists():
        st.info("Prepare data/liu_ice/guard_bundle.joblib to enable the measured replay.")
    else:
        scenario_id = st.selectbox(
            "Held-out fault replay",
            list(LIU_SCENARIOS),
            format_func=lambda key: f"{LIU_SCENARIOS[key]['name']} · {LIU_SCENARIOS[key]['code']}",
            key="liu_scenario",
        )
        replay = build_guard_replay(measured_guard_bundle(), scenario_id)
        cursor = st.slider("Replay frame", 0, len(replay.frame) - 1, min(20, len(replay.frame) - 1), key="liu_cursor")
        row = replay.frame.iloc[cursor]
        detector = {"Intercooler_pressure": "f_pic", "intake_manifold_pressure": "f_pim", "air_mass_flow": "f_waf"}[replay.scenario["fault_sensor"]]
        confirmed = bool(replay.confirmed.iloc[cursor][detector])
        evidence_now = float(replay.evidence.iloc[cursor][detector])
        health_now = float(replay.health.iloc[cursor])
        measured_rpm = float(row["engine_speed__mean"] * 60 / (2 * np.pi))
        measured_load = float(np.clip(row["throttle_position__mean"], 0, 100))

        st.markdown(
            toolbar("LIU-ICE", float(row["window_end_s"]), "sensor_drift" if confirmed else "healthy", 0,
                    float(replay.frame["window_end_s"].iloc[-1])),
            unsafe_allow_html=True,
        )
        kpis = st.columns(4)
        measured_cards = (
            ("Measured engine speed", f"{measured_rpm:,.0f}", "RPM", "LiU-ICE bench channel", "#d8aa32"),
            ("Guard evidence", f"{evidence_now:.2f}", "×", "alarm limit = 1.00×", "#d78b80" if confirmed else "#d8aa32"),
            ("Health index", f"{health_now:.0f}", "/ 100", "residual-derived indicator", "#f4c95d"),
            ("Guard decision", "CONFIRMED" if confirmed else "MONITORING", "", replay.scenario["code"], "#d78b80" if confirmed else "#d8aa32"),
        )
        for column, card in zip(kpis, measured_cards):
            column.markdown(metric_card(*card), unsafe_allow_html=True)

        history = replay.evidence.iloc[: cursor + 1]
        evidence_fig = go.Figure()
        evidence_colours = {"f_pic": "#d8b453", "f_pim": "#9ba9ba", "f_waf": "#a1b3a2"}
        evidence_names = {"f_pic": "Intercooler pressure", "f_pim": "Manifold pressure", "f_waf": "Air mass flow"}
        for code in history:
            evidence_fig.add_trace(go.Scatter(
                x=replay.frame.loc[:cursor, "window_end_s"], y=history[code], name=evidence_names[code],
                line={"color": evidence_colours[code], "width": 2},
            ))
        evidence_fig.add_hline(y=1.0, line={"color": "#d78b80", "dash": "dash"}, annotation_text="alarm limit")
        evidence_fig.add_vline(x=replay.injection_time, line={"color": "#f4c95d", "dash": "dot"}, annotation_text="injected bias")
        evidence_fig.update_layout(
            title={"text": "MEASURED REDUNDANT-SENSOR EVIDENCE", "font": {"size": 13, "family": "Arial"}},
            paper_bgcolor="rgba(7,23,34,.75)", plot_bgcolor="#151614", font={"color": "#b3b3aa", "family": "Arial"},
            height=360, margin={"l": 35, "r": 18, "t": 54, "b": 38}, legend={"orientation": "h", "y": 1.14},
            xaxis={"title": "run time · seconds", "gridcolor": "#30312d"},
            yaxis={"title": "residual / learned healthy limit", "gridcolor": "#30312d", "rangemode": "tozero"},
        )
        left, right = st.columns([1.72, 1.0], gap="small")
        with left:
            st.plotly_chart(evidence_fig, width="stretch", config={"displayModeBar": False})
            st.markdown(sensor_grid(replay.sensor_readings(cursor)), unsafe_allow_html=True)
        with right:
            st.markdown(engine_panel("sensor_drift" if confirmed else "healthy", max(0.0, (100 - health_now) / 34), measured_rpm, measured_load), unsafe_allow_html=True)
            state_text = "FAULT CONFIRMED" if confirmed else "CONSISTENCY CHECK ACTIVE"
            st.markdown(
                f'<div class="et-demo"><b>{state_text}</b><span>Decision requires 2 of 3 consecutive five-second windows and at least 1.25× evidence in the current window.</span></div>',
                unsafe_allow_html=True,
            )
