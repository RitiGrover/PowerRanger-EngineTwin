"""EngineTwin dashboard: one app for live diagnosis, signatures, efficiency, health, data and results.

Run with:  streamlit run app.py
Everything it shows is computed from models and tables saved in cache/ (build them with the scripts in scripts/);
nothing is trained while the dashboard runs."""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from enginetwin import dashboard as dash
from enginetwin.diagnosis import Diagnoser
from enginetwin.efficiency_map import efficiency_map
from enginetwin.efficiency_model import KIA_SOUL_MAX_TORQUE_NM
from enginetwin.fault_features import refine_rpm
from enginetwin.health import time_to_service
from enginetwin.latency import simulate_onset
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

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
SPEC = VehicleSpec()
STREAM_SECONDS, ONSET_S, WINDOW_S = 8.0, 3.0, 2.0

st.set_page_config(page_title="EngineTwin", layout="wide")


# ---------------------------------------------------------------- cached resources (loaded once, never trained here)
@st.cache_resource
def diagnoser() -> Diagnoser:
    return Diagnoser.load(CACHE)


@st.cache_resource
def efficiency_model():
    return pickle.load(open(CACHE / "model_efficiency_gbm.pkl", "rb"))


@st.cache_data
def ocslab_cells() -> pd.DataFrame:
    return efficiency_map(load_ocslab(next((ROOT / "data" / "osclab").glob("*.csv"))))


@st.cache_data
def make_stream(fault, severity, rpm, load, snr_db, seed):
    return simulate_onset(fault, severity, onset_s=ONSET_S, duration_s=STREAM_SECONDS, rpm=rpm, seed=seed, load=load, snr_db=snr_db)


@st.cache_data
def diagnose_at(fault, severity, rpm, load, snr_db, seed, end_s):
    window = make_stream(fault, severity, rpm, load, snr_db, seed).window(end_s, WINDOW_S)
    return diagnoser().diagnose(window)


@st.cache_data
def read_json(name):
    path = CACHE / name
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def trip_replay(name, fault, severity, loss_rates):
    """Healthy replays come from the saved tables; with a fault the first 900 seconds are replayed live with its loss."""
    source = "ocslab" if name == "Ocslab trip H-0 (held-out driver)" else "ved"
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


# ---------------------------------------------------------------- sidebar: controls
missing = [n for n in ("model_cnn.pkl", "model_softmax.pkl", "model_autoencoder.pkl", "model_severity.pkl", "baseline.pkl", "model_efficiency_gbm.pkl") if not (CACHE / n).exists()]
if missing:
    st.error("Models are missing from cache/: " + ", ".join(missing) + ". Build them with the scripts in scripts/ (see README).")
    st.stop()

with st.sidebar:
    st.header("Controls")
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
    trip_name = st.selectbox("Replay a real trip", ["Ocslab trip H-0 (held-out driver)", "VED trip 488-438 (held-out car)"], key="trip")

mode = dash.drive_mode_point(drive_mode)
sev = 0.0 if injected == "healthy" else float(severity)
params = (injected, sev, mode["rpm"], mode["load"], float(snr_db), int(seed))
stream = make_stream(*params)

st.title("EngineTwin")
st.caption("Engine fault signatures and efficiency sweet spot from multi-sensor data. Sound and vibration are simulated; "
           "every chart is tagged Real, Simulated or Mixed.")
tabs = st.tabs(["Live engine", "Signatures", "Diagnosis", "Efficiency and sweet spot", "Health", "Data explorer", "Real recording", "Results"])
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


# ---------------------------------------------------------------- Live engine
with tabs[0]:
    st.subheader("Live engine")
    st.caption(f"Simulated engine in {drive_mode} mode. The fault, if any, starts at {ONSET_S:.0f} s. Playback position: {end_s:.1f} s.")
    banner, gauge_row, wave_row = st.empty(), st.empty(), st.empty()

    def draw_live(res, win, sound_chunk, vibration_chunk):
        with banner.container():
            status_banner(res)
        with gauge_row.container():
            cols = st.columns(5)
            readings = (("Engine speed", win.obd["engine_speed"], 0, 6000, "rpm"), ("Load", win.obd["calculated_load_value"], 0, 100, "%"),
                        ("Coolant", win.obd["engine_coolant_temperature"], 40, 120, "C"), ("Timing advance", win.obd["timing_advance"], -10, 50, "deg"),
                        ("Total fuel trim", win.obd["short_term_fuel_trim_bank1"] + win.obd["long_term_fuel_trim_bank1"], -25, 25, "%"))
            for col, (name, value, low, high, unit) in zip(cols, readings):
                col.plotly_chart(dash.gauge_figure(name, float(value), low, high, unit, "mixed"), width="stretch")
        with wave_row.container():
            import plotly.graph_objects as go

            t = np.arange(len(sound_chunk)) / win.sound_fs
            fig = go.Figure([go.Scatter(x=t, y=sound_chunk, name="Sound"), go.Scatter(x=t, y=vibration_chunk, name="Vibration")])
            fig.update_layout(title=dash.tagged_title("Latest half-second chunk", "simulated"), xaxis_title="Time (s)", height=260)
            st.plotly_chart(fig, width="stretch")

    stop = int(round(end_s * stream.fs))
    draw_live(result, window, stream.sound[stop - stream.fs // 2 : stop], stream.vibration[stop - stream.fs // 2 : stop])
    if st.button("Stream live (half-second chunks)", key="stream"):
        for t_end, win, sound_chunk, vibration_chunk in rolling_windows(stream, 0.5, WINDOW_S):
            draw_live(diagnose_at(*params, round(t_end, 1)), win, sound_chunk, vibration_chunk)
            time.sleep(0.4)
    if not has_obd(window):
        st.warning("No OBD readings.")
    st.caption(f"Estimated noise level: {result.noise_db:.0f} dB noise to signal. "
               + ("The autoencoder finds this window unfamiliar." if result.anomaly else "The autoencoder finds this window familiar."))

# ---------------------------------------------------------------- Signatures
with tabs[1]:
    st.subheader("Signatures")
    spec = window_spectrogram(window)
    left, right = st.columns(2)
    left.plotly_chart(dash.spectrogram_figure(spec[0], WINDOW_S, "Sound", "simulated"), width="stretch")
    right.plotly_chart(dash.spectrogram_figure(spec[1], WINDOW_S, "Vibration", "simulated"), width="stretch")
    orders, amps = order_spectrum(window.sound, window.sound_fs, rpm_now, max_order=8.0)
    left.plotly_chart(dash.order_spectrum_figure(orders, amps, "simulated"), width="stretch")
    env_f, env_a = envelope_spectrum(window.vibration, window.vibration_fs, band=(2500, 3500))
    keep = env_f <= 300
    right.plotly_chart(dash.envelope_figure(env_f[keep], env_a[keep], rpm_now, "simulated"), width="stretch")
    n_samples = int(12 * 30.0 / rpm_now * window.vibration_fs)
    energy = per_cylinder_energy(window.vibration[:n_samples], window.vibration_fs, rpm_now)
    left.plotly_chart(dash.cylinder_figure(energy, "simulated"), width="stretch")
    right.info("A misfire adds a line at order 0.5 and leaves one cylinder with little firing energy. Knock adds short bursts near "
               "6.5 kHz after each firing. A worn valve ticks near 3 kHz once per two revolutions. These signatures are assumptions, "
               "not measured from real faults.")

# ---------------------------------------------------------------- Diagnosis
with tabs[2]:
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
        st.caption(f"Explainable baseline (softmax on hand-made features) says: {result.softmax_label}. "
                   f"Autoencoder: {'unfamiliar' if result.anomaly else 'familiar'} (score {result.anomaly_score:.2f}, alarm level {result.anomaly_threshold:.2f}).")
    detected = [f for f in result.probabilities if f in result.label]
    if detected:
        fault = st.selectbox("Show what triggered", detected, key="heat_fault")
        from enginetwin.spectrograms import standardise_spectrogram

        heat = diagnoser().heatmap(window, fault)
        st.plotly_chart(dash.heat_overlay_figure(standardise_spectrogram(spec)[0], heat, "simulated"), width="stretch")
        st.caption("Heat map: how much the fault probability drops when each patch of the spectrogram is hidden (sound and vibration together).")
    else:
        st.info("No fault detected, so there is no trigger to show.")

# ---------------------------------------------------------------- Efficiency and sweet spot
with tabs[3]:
    st.subheader("Efficiency and sweet spot")
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
    st.plotly_chart(dash.efficiency_map_figure(cells, (rec["actual"]["rpm"], rec["actual"]["torque_nm"]), (rec["rpm"], rec["torque_nm"]), "real"), width="stretch")
    st.caption("The map is measured Ocslab data (Kia Soul, 10 drivers). The points are where the engine runs now (white) and where the optimiser would run it (gold) "
               "for the same power, using an efficiency model learned from Ocslab. The Ocslab fuel unit is inferred as microlitres per second, "
               "and absolute efficiencies are about half of EPA's, so read the shape and the gains rather than the absolute level.")
    st.markdown("**Replay of a real trip: actual against optimised**")
    table = trip_replay(trip_name, injected if injected != "healthy" else "healthy", sev, tuple(loss_rates.items()))
    actual_l, optimised_l = summarise(table)
    cols = st.columns(3)
    cols[0].metric("Fuel actually used", f"{actual_l:.3f} L")
    cols[1].metric("Fuel with optimised gears", f"{optimised_l:.3f} L")
    cols[2].metric("Saved", f"{actual_l - optimised_l:.3f} L", f"Rs {(actual_l - optimised_l) * 105:.1f}")
    st.plotly_chart(dash.series_figure(table, "mixed"), width="stretch")
    st.caption("Mixed: speed, rpm and fuel are recorded values; the optimised series is modelled. "
               + ("The VED trip has no torque, so its power comes from a vehicle model and the efficiency model did not validate on VED: treat that saving as an illustration."
                  if trip_name.startswith("VED") else "The saving on this held-out Ocslab trip is small: the car's automatic already picks gears close to what the model prefers."))

# ---------------------------------------------------------------- Health
with tabs[4]:
    st.subheader("Health")
    health = read_json("results_health_autoencoder.json")
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
        st.caption("A simulated valve wears from new to fully worn over 60 days (the wear curve is an assumption). Health is estimated from the vibration "
                   f"alone (error {health['wear_regressor_mae_test_same']:.3f} in severity). The forecast extends the last 14 days in a straight line, so "
                   "it is early or late while wear accelerates: it only becomes accurate in the last third.")
    else:
        st.info("Run scripts/train_health_and_autoencoder.py to build the health series.")

# ---------------------------------------------------------------- Data explorer
with tabs[5]:
    st.subheader("Data explorer")
    st.dataframe(dash.data_provenance(), hide_index=True, width="stretch")
    st.markdown("**Training and demonstration data are different.** The efficiency model learns from 8 Ocslab drivers and is shown on the other 2 and on VED cars. "
                "The fault models learn from simulated training windows and are shown on freshly simulated streams with other seeds and other noise.")
    st.markdown("**What is real and what is simulated.** The efficiency map, trip features and OBD calibration are real. Sound and vibration are simulated from assumed "
                "fault signatures, so scores on simulated test sets show that the models find the patterns we put in, not accuracy on a real engine.")

# ---------------------------------------------------------------- Real recording
with tabs[6]:
    st.subheader("Real recording")
    st.warning("The fault models were trained on simulated faults only. A real recording may sound different from anything they have seen, "
               "so treat the result as a demonstration of the pipeline, not a diagnosis.")
    wav = st.file_uploader("Engine recording (WAV)", type=["wav"], key="wav")
    obd_file = st.file_uploader("OBD log (CSV, optional)", type=["csv"], key="obd_csv")
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
with tabs[7]:
    st.subheader("Results")
    st.warning("Scores below are on simulated sound and vibration. They do not show real-engine accuracy.")
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
