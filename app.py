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
from enginetwin.signals import envelope_spectrum, order_spectrum, per_cylinder_energy
from enginetwin.spectrograms import window_spectrogram
from enginetwin.streaming import rolling_windows
from enginetwin.vehicle import VehicleSpec, engine_point, wheel_power_w
from enginetwin.ui import COCKPIT_CSS, engine_panel, event_log, hero, metric_card, scroll_engine_simulation, sensor_grid, toolbar
from enginetwin.theme import THEME_CSS, brand_block, register_plotly_theme
from enginetwin.scenes import engine_map_html, engine_viewer_html, landing_html
from enginetwin.sensor_map import build_payload

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
LIU_GUARD_BUNDLE = ROOT / "data" / "liu_ice" / "guard_bundle.joblib"
SPEC = VehicleSpec()
STREAM_SECONDS, ONSET_S, WINDOW_S = 8.0, 3.0, 2.0
REQUIRED_MODELS = ("model_cnn.pkl", "model_softmax.pkl", "model_autoencoder.pkl", "model_severity.pkl", "baseline.pkl", "model_efficiency_gbm.pkl")
MISSING_MODELS = [name for name in REQUIRED_MODELS if not (CACHE / name).exists()]
DEMO_MODE = bool(MISSING_MODELS)

EMPTY_FAVICON = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'/%3E"
st.set_page_config(page_title="Power Ranger", page_icon=EMPTY_FAVICON, layout="wide")
register_plotly_theme()
if not getattr(st.plotly_chart, "_enginetwin", False):
    # Let the shared Plotly template style every chart instead of Streamlit's own chart theme.
    _plotly_chart = st.plotly_chart

    def _themed_plotly_chart(*args, **kwargs):
        return _plotly_chart(*args, **{"theme": None, **kwargs})

    _themed_plotly_chart._enginetwin = True
    st.plotly_chart = _themed_plotly_chart
st.markdown(COCKPIT_CSS, unsafe_allow_html=True)
st.markdown(THEME_CSS, unsafe_allow_html=True)

# The product-configurator opening is intentionally isolated from the analytical
# dashboard. This keeps the first impression quiet and avoids loading models
# until the visitor asks to enter the live twin.
if "entered_engine_twin" not in st.session_state:
    st.session_state.entered_engine_twin = False
if st.query_params.get("view") == "cockpit":
    st.session_state.entered_engine_twin = True
LANDING_CSS = """<style>
.stApp:has(.et-landing) [data-testid="stHeader"], .stApp:has(.et-landing) [data-testid="stSidebar"],
.stApp:has(.et-landing) [data-testid="stSidebarCollapsedControl"], .stApp:has(.et-landing) [data-testid="stToolbar"] { display:none !important; }
.stApp:has(.et-landing)::before { display:none; }
.stApp:has(.et-landing) .block-container { max-width:none; padding:0; }
.stApp:has(.et-landing) iframe { position:fixed; inset:0; width:100vw; height:100vh; border:0; z-index:5; }
.stApp:has(.et-landing) [data-testid="stElementContainer"]:has(.et-landing) { display:none; }
.stApp:has(.et-landing) [data-testid="stButton"] { position:fixed; right:4vw; bottom:24px; z-index:50; width:auto; }
.stApp:has(.et-landing) [data-testid="stButton"] button { height:42px; min-width:184px; padding:0 22px; border-radius:6px !important; background:#bd3a33 !important; color:#fff5f3 !important; border-color:#d05249 !important; font-weight:600; font-size:.82rem; box-shadow:0 9px 28px rgba(153,35,29,.20) !important; }
.stApp:has(.et-landing) [data-testid="stButton"] button:hover { background:#d04a42 !important; color:#ffffff !important; border-color:#e06a61 !important; }
</style><div class="et-landing"></div>"""
if not st.session_state.entered_engine_twin:
    st.markdown(LANDING_CSS, unsafe_allow_html=True)
    st.iframe(landing_html(), height=800)
    if st.button("Enter live twin  →", key="enter_engine_twin"):
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


@st.cache_data(show_spinner="Simulating the seven engine states")
def engine_map_payload(rpm, load, severity, snr_db, seed, demo):
    """Runs the existing simulator, feature extractor and diagnoser once per fault label; display data only."""
    return build_payload(lambda window: diagnoser().diagnose(window), rpm, load, severity, snr_db, seed, demo)


def summarise(table):
    litres = lambda col: float((table[col] * table["time_s"].diff().fillna(1.0)).sum() / 3600)
    actual, optimised = litres("actual_fuel_l_h"), litres("opt_fuel_l_h")
    return actual, optimised


SECTIONS = ["Live engine", "Engine map", "Signatures", "Diagnosis", "Sweet spot", "Health", "Data explorer", "Real recording", "Results", "Measured guard", "Synthetic engine"]
NAV_GROUPS = (
    ("Monitor", ("Live engine", "Engine map")),
    ("Analysis", ("Signatures", "Diagnosis", "Synthetic engine")),
    ("Insights", ("Sweet spot", "Health")),
    ("Data & evidence", ("Real recording", "Measured guard", "Results", "Data explorer")),
)
# Keep page-specific inputs when their widgets are temporarily off screen.
for input_key in list(st.session_state):
    if input_key in {"guard_enabled", "triggered_failure", "health_day", "heat_fault", "confusion", "liu_scenario", "liu_cursor"} or input_key.startswith("manual_bias_"):
        st.session_state[input_key] = st.session_state[input_key]

with st.sidebar:
    st.markdown(brand_block(), unsafe_allow_html=True)
    if st.session_state.get("workspace_section") not in SECTIONS:
        st.session_state.workspace_section = SECTIONS[0]
    item_number = 0
    for group_name, pages in NAV_GROUPS:
        st.markdown(f'<div class="et-nav-group">{group_name}</div>', unsafe_allow_html=True)
        for page in pages:
            item_number += 1
            key = "nav_" + page.lower().replace(" ", "_").replace("&", "and")
            selected = st.session_state.workspace_section == page
            if st.button(
                f"{item_number:02d}   {page}",
                key=key,
                type="primary" if selected else "secondary",
                use_container_width=True,
            ):
                st.session_state.workspace_section = page
                st.rerun()
    section = st.session_state.workspace_section
    st.divider()

# ---------------------------------------------------------------- sidebar: controls
if DEMO_MODE:
    st.sidebar.caption("Demo data: the synthetic sections use demo models until trained cache artifacts exist. Live engine and Measured guard use the supplied LiU-ICE replay.")

with st.sidebar.expander("Simulation settings", expanded=False):
    st.caption("Controls for synthetic analysis and trip replay.")
    drive_mode = st.selectbox("Drive mode", list(dash.DRIVE_MODES), index=1, key="drive_mode")
    injected = st.selectbox("Inject a fault", FAULTS, key="fault", help="The simulated engine develops this fault 3 s into the stream.")
    severity = st.slider("Fault severity", 0.0, 1.0, 0.8, 0.05, key="severity", disabled=injected == "healthy")
    snr_db = st.slider("Noise level (signal-to-noise, dB)", 0, 30, 20, key="snr", help="Lower is noisier. Models were trained on 10 to 25 dB.")
    seed = st.number_input("Random seed", 0, 9999, 7, key="seed")
    st.subheader("Fault loss assumptions")
    st.caption("Share of efficiency each fault removes at full severity. These are assumptions, not measurements.")
    loss_rates = {
        "misfire": st.slider("Misfire loss (%)", 0, 60, int(DEFAULT_LOSS_PER_SEVERITY["misfire"] * 100), key="loss_misfire") / 100,
        "knock": st.slider("Knock loss (%)", 0, 60, int(DEFAULT_LOSS_PER_SEVERITY["knock"] * 100), key="loss_knock") / 100,
        "valve_wear": st.slider("Valve wear loss (%)", 0, 60, int(DEFAULT_LOSS_PER_SEVERITY["valve_wear"] * 100), key="loss_valve") / 100,
    }
    trip_options = (["Synthetic urban trip", "Synthetic highway trip"] if DEMO_MODE else
                    ["Ocslab trip H-0 (held-out driver)", "VED trip 488-438 (held-out car)"])
    trip_name = st.selectbox("Replay scenario" if DEMO_MODE else "Replay a real trip", trip_options, key="trip")

mode = dash.drive_mode_point(drive_mode)
sev = 0.0 if injected == "healthy" else float(severity)
params = (injected, sev, mode["rpm"], mode["load"], float(snr_db), int(seed))
stream = make_stream(*params)

steps = [round(WINDOW_S + 0.5 * k, 1) for k in range(int((STREAM_SECONDS - WINDOW_S) / 0.5) + 1)]
end_s = st.sidebar.select_slider("Playback position (s)", options=steps, value=5.0, key="position")
result = diagnose_at(*params, end_s)
window = stream.window(end_s, WINDOW_S)
rpm_now = refine_rpm(window.sound, window.sound_fs, window.obd["engine_speed"])


def status_banner(res):
    if res.status == "Healthy":
        st.success(f"Healthy (confidence {res.confidence:.0%})")
    else:
        st.error(f"Fault: {res.label.replace('+', ' and ').replace('_', ' ')} (confidence {res.confidence:.0%})")


# ---------------------------------------------------------------- Engine map
if section == "Engine map":
    st.markdown('<div class="et-kicker">Engine map</div>', unsafe_allow_html=True)
    st.subheader("Which sensor sees which fault")
    st.caption("Pick a fault to see the parts it acts on, how each sensor reading changes, and how those changes become the diagnosis. "
               "The operating point follows the drive mode in Simulation settings.")
    slider_col, _spacer = st.columns([1, 2])
    with slider_col:
        map_severity = st.slider("Fault severity for this view", 0.2, 1.0, float(severity) if severity else 0.8, 0.05, key="map_severity")
    payload = engine_map_payload(float(mode["rpm"]), float(mode["load"]), float(map_severity), float(snr_db), int(seed), DEMO_MODE)
    st.iframe(engine_map_html(payload), height="content")

# ---------------------------------------------------------------- Live engine
if section == "Live engine":
    render_live()

if section == "Synthetic engine":
    trim = window.obd["short_term_fuel_trim_bank1"] + window.obd["long_term_fuel_trim_bank1"]
    kpis = st.columns(4)
    kpi_data = (
        ("Engine speed", f"{rpm_now:,.0f}", "RPM", "Crankshaft angular velocity", "#a9c4d4"),
        ("Calculated load", f"{window.obd['calculated_load_value']:.1f}", "%", "Combustion demand", "#a9c4d4"),
        ("Coolant temperature", f"{window.obd['engine_coolant_temperature']:.1f}", "°C", "Thermal operating state", "#d49a6a"),
        ("AI guard state", result.status.upper(), "", f"{result.confidence:.0%} confidence", "#a9c4d4" if result.status == "Healthy" else "#d78b80"),
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
    trace_fig.add_trace(go.Scatter(x=chart_t, y=norm(sound_trace, 68), name="Acoustic", line={"color": "#a9c4d4", "width": 1.6}))
    trace_fig.add_trace(go.Scatter(x=chart_t, y=norm(vibration_trace, 38), name="Vibration", line={"color": "#a5aab0", "width": 1.5}))
    trace_fig.add_trace(go.Scatter(x=chart_t, y=np.full_like(chart_t, window.obd["calculated_load_value"]), name="Engine load", line={"color": "#a1b3a2", "width": 1.4}))
    if result.status != "Healthy":
        trace_fig.add_trace(go.Scatter(x=chart_t, y=np.full_like(chart_t, 8), name="Fault evidence", line={"color": "#d78b80", "width": 2, "dash": "dash"}))
    trace_fig.update_layout(
        title={"text": "SYNCHRONIZED SENSOR STREAMS & FAULT EVIDENCE", "font": {"size": 13, "family": "Helvetica Neue, Helvetica, Arial"}},
        paper_bgcolor="rgba(19,20,22,.75)", plot_bgcolor="#101214", font={"color": "#9aa0a6", "family": "Helvetica Neue, Helvetica, Arial"},
        height=388, margin={"l": 35, "r": 18, "t": 54, "b": 36}, legend={"orientation": "h", "y": 1.12, "x": .46},
        xaxis={"title": "Replay time · seconds", "gridcolor": "#26292d"}, yaxis={"title": "normalized signal", "gridcolor": "#26292d", "range": [0, 100]},
    )
    left, right = st.columns([1.72, 1.0], gap="small")
    with left:
        st.plotly_chart(trace_fig, width="stretch", config={"displayModeBar": False})
        # Engine speed, load, coolant and playback already appear in the cards above and the sidebar.
        readings = (
            ("Air mass flow", f"{window.obd['maf']:.2f}", "g/s", "NOMINAL", "#a9c4d4"),
            ("Timing advance", f"{window.obd['timing_advance']:.1f}", "deg", "STREAMING", "#a9c4d4"),
            ("Fuel trim", f"{trim:+.1f}", "%", "WATCH" if abs(trim) > 8 else "NOMINAL", "#d49a6a"),
            ("Noise floor", f"{snr_db}", "dB", "SYNCHRONIZED", "#a9c4d4"),
            ("Fault evidence", f"{max(result.probabilities.values()):.0%}", "", result.label.upper(), "#d78b80" if result.status != "Healthy" else "#a9c4d4"),
        )
        st.markdown(sensor_grid(readings), unsafe_allow_html=True)
    with right:
        st.iframe(engine_viewer_html(result.label, float(np.mean(list(result.severities.values()))) if result.severities else 0.0,
                                           rpm_now, window.obd["calculated_load_value"]), height=420)
        st.markdown(event_log(result.label, result.confidence, end_s), unsafe_allow_html=True)
    if not has_obd(window):
        st.warning("No OBD readings.")

# ---------------------------------------------------------------- Signatures
if section == "Signatures":
    st.subheader("Signatures")
    spec = window_spectrogram(window)
    compact = lambda fig: fig.update_layout(height=300, margin={"l": 44, "r": 16, "t": 48, "b": 40})
    col_a, col_b, col_c = st.columns(3)
    col_a.plotly_chart(compact(dash.spectrogram_figure(spec[0], WINDOW_S, "Sound", "simulated")), width="stretch")
    col_b.plotly_chart(compact(dash.spectrogram_figure(spec[1], WINDOW_S, "Vibration", "simulated")), width="stretch")
    orders, amps = order_spectrum(window.sound, window.sound_fs, rpm_now, max_order=8.0)
    col_c.plotly_chart(compact(dash.order_spectrum_figure(orders, amps, "simulated")), width="stretch")
    env_f, env_a = envelope_spectrum(window.vibration, window.vibration_fs, band=(2500, 3500))
    keep = env_f <= 300
    col_a.plotly_chart(compact(dash.envelope_figure(env_f[keep], env_a[keep], rpm_now, "simulated")), width="stretch")
    n_samples = int(12 * 30.0 / rpm_now * window.vibration_fs)
    energy = per_cylinder_energy(window.vibration[:n_samples], window.vibration_fs, rpm_now)
    col_b.plotly_chart(compact(dash.cylinder_figure(energy, "simulated")), width="stretch")
    col_c.info("A misfire adds a line at order 0.5 and leaves one cylinder with little firing energy. Knock adds short bursts near "
               "6.5 kHz after each firing. A worn valve ticks near 3 kHz once per two revolutions. These signatures are assumptions, "
               "not measured from real faults.")

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
    tab_map, tab_replay = st.tabs(["Efficiency map", "Trip replay"])
    with tab_map:
        map_source = "simulated" if DEMO_MODE else "real"
        st.plotly_chart(dash.efficiency_map_figure(cells, (rec["actual"]["rpm"], rec["actual"]["torque_nm"]), (rec["rpm"], rec["torque_nm"]), map_source).update_layout(height=440), width="stretch")
        if DEMO_MODE:
            st.caption("Demo efficiency surface: the current point is white and the recommended point is gold. Replace demo mode with trained artifacts before reporting these values.")
        else:
            st.caption("The map is measured Ocslab data (Kia Soul, 10 drivers). The points are where the engine runs now (white) and where the optimiser would run it (gold) "
                       "for the same power, using an efficiency model learned from Ocslab. The Ocslab fuel unit is inferred as microlitres per second, "
                       "and absolute efficiencies are about half of EPA's, so read the shape and the gains rather than the absolute level.")
    with tab_replay:
        st.markdown("**Scenario replay: actual against optimised**" if DEMO_MODE else "**Replay of a real trip: actual against optimised**")
        table = trip_replay(trip_name, injected if injected != "healthy" else "healthy", sev, tuple(loss_rates.items()))
        actual_l, optimised_l = summarise(table)
        cols = st.columns(3)
        cols[0].metric("Fuel actually used", f"{actual_l:.3f} L")
        cols[1].metric("Fuel with optimised gears", f"{optimised_l:.3f} L")
        cols[2].metric("Saved", f"{actual_l - optimised_l:.3f} L", f"Rs {(actual_l - optimised_l) * 105:.1f}")
        st.plotly_chart(dash.series_figure(table, "mixed").update_layout(height=520), width="stretch")
        if DEMO_MODE:
            st.caption("Simulated scenario: both series are generated for interface testing. The values are illustrative, not measured savings.")
        else:
            st.caption("Mixed: speed, rpm and fuel are recorded values; the optimised series is modelled. "
                       + ("The VED trip has no torque, so its power comes from a vehicle model and the efficiency model did not validate on VED: treat that saving as an illustration."
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
                       f"alone (error {health['wear_regressor_mae_test_same']:.3f} in severity). The forecast extends the last 14 days in a straight line, so "
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
    st.subheader("Real recording")
    if DEMO_MODE:
        st.info("Demo mode can visualise an uploaded WAV, but it does not classify real recordings. Build the trained artifacts for the experimental pipeline.")
    else:
        st.warning("The fault models were trained on simulated faults only. A real recording may sound different from anything they have seen, "
                   "so treat the result as a demonstration of the pipeline, not a diagnosis.")
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

        kpis = st.columns(4)
        measured_cards = (
            ("Measured engine speed", f"{measured_rpm:,.0f}", "RPM", "LiU-ICE bench channel", "#a9c4d4"),
            ("Guard evidence", f"{evidence_now:.2f}", "×", "alarm limit = 1.00×", "#d78b80" if confirmed else "#a9c4d4"),
            ("Health index", f"{health_now:.0f}", "/ 100", "residual-derived indicator", "#c5d6a0"),
            ("Guard decision", "CONFIRMED" if confirmed else "MONITORING", "", replay.scenario["code"], "#d78b80" if confirmed else "#a9c4d4"),
        )
        for column, card in zip(kpis, measured_cards):
            column.markdown(metric_card(*card), unsafe_allow_html=True)

        history = replay.evidence.iloc[: cursor + 1]
        evidence_fig = go.Figure()
        evidence_colours = {"f_pic": "#a9c4d4", "f_pim": "#a5aab0", "f_waf": "#a1b3a2"}
        evidence_names = {"f_pic": "Intercooler pressure", "f_pim": "Manifold pressure", "f_waf": "Air mass flow"}
        for code in history:
            evidence_fig.add_trace(go.Scatter(
                x=replay.frame.loc[:cursor, "window_end_s"], y=history[code], name=evidence_names[code],
                line={"color": evidence_colours[code], "width": 2},
            ))
        evidence_fig.add_hline(y=1.0, line={"color": "#d78b80", "dash": "dash"}, annotation_text="alarm limit")
        evidence_fig.add_vline(x=replay.injection_time, line={"color": "#c5d6a0", "dash": "dot"}, annotation_text="injected bias")
        evidence_fig.update_layout(
            title={"text": "MEASURED REDUNDANT-SENSOR EVIDENCE", "font": {"size": 13, "family": "Helvetica Neue, Helvetica, Arial"}},
            paper_bgcolor="rgba(19,20,22,.75)", plot_bgcolor="#101214", font={"color": "#9aa0a6", "family": "Helvetica Neue, Helvetica, Arial"},
            height=360, margin={"l": 35, "r": 18, "t": 54, "b": 38}, legend={"orientation": "h", "y": 1.14},
            xaxis={"title": "run time · seconds", "gridcolor": "#26292d"},
            yaxis={"title": "residual / learned healthy limit", "gridcolor": "#26292d", "rangemode": "tozero"},
        )
        left, right = st.columns([1.72, 1.0], gap="small")
        with left:
            st.plotly_chart(evidence_fig, width="stretch", config={"displayModeBar": False})
            st.markdown(sensor_grid(replay.sensor_readings(cursor)), unsafe_allow_html=True)
        with right:
            st.iframe(engine_viewer_html("sensor_drift" if confirmed else "healthy", max(0.0, (100 - health_now) / 34), measured_rpm, measured_load), height=420)
            st.caption("Decision requires 2 of 3 consecutive five-second windows and at least 1.25× evidence in the current window.")
