# PowerRanger-EngineTwin
Engine fault signatures and efficiency sweet spot finder from multi-sensor engine data

## What is real and what is simulated

The sound and vibration data is simulated. No real engine audio is used.

| Part | Source |
| --- | --- |
| OBD readings (rpm, load, timing advance, fuel trims, MAF) | Calibrated to real Kia Soul rows from the Ocslab data |
| Efficiency map and trip features | Real data (Ocslab, VED, EPA) |
| Sound and vibration | Simulated from assumed signatures |

The assumed signatures are:

- Misfire: one cylinder fires weakly, which adds a line at order 0.5.
- Knock: short bursts near 6.5 kHz just after firing.
- Valve wear: a tick near 3 kHz, once per two revolutions for each worn valve (one or two valves).

These are textbook style assumptions. They have not been checked against real fault recordings.

Because of this, a high score on the simulated test splits only shows that a model finds the patterns we put in. It does not show the same accuracy on a real engine. Results on simulated and real windows must be reported separately. Every window carries a `source` tag ("simulated" or "real") for this reason.

The test splits made by `make_splits` are meant to check robustness, not realism:

- `test_unseen_rpm` uses engine speeds never seen in training.
- `test_noisy` uses the training speeds with much more background noise.

## Checking against real recordings

If real fault audio becomes available, run the same signal tools on it:

- `order_spectrum` or `half_order_ratio` for misfire.
- `band_energy_fraction` in the knock band for knock.
- `envelope_spectrum` for valve ticks.

If the lines sit at different frequencies, change the constants at the top of `src/enginetwin/simulator.py` and regenerate the data.

## Simulated training data

`make_labelled_splits` (see `scripts/make_labelled_dataset.py`) writes five cached splits of 2 s windows to `cache/`: train, validation, test at the same conditions, test at unseen speeds (3000 to 5000 rpm) and test at heavy noise (5 dB signal to noise, against 15 to 25 dB elsewhere). Window i has the same speed, load and noise level in every fault class, so the operating point cannot give the label away.

The efficiency truth for each window is the healthy efficiency (nearest cell of the real Ocslab map) less an assumed fault loss: misfire 25%, knock 8% and valve wear 3% of efficiency at full severity (`enginetwin.losses`). These loss rates are assumptions, not measurements, and the dashboard lets the user change them. `wear_series` simulates one valve wearing from new to fully worn over many days.

## Running it

```
python -m venv .venv
.venv\Scripts\pip install -e .[dev,dl]
```

Build the cached data and models once (nothing is trained while the dashboard runs). Run these from the project folder with `python -I`:

1. `scripts/make_labelled_dataset.py` writes the simulated splits to `cache/`.
2. `scripts/train_classical.py` (SVM and softmax), `scripts/train_obd_ann.py`, `scripts/train_cnn.py`, `scripts/train_efficiency.py` and `scripts/train_health_and_autoencoder.py` fit and score the models.
3. `scripts/prepare_dashboard.py` bundles the diagnoser, `scripts/replay_demo.py` replays one held-out Ocslab trip and one VED trip, and `scripts/measure_latency.py` writes the latency and results tables.

Then start the dashboard with `streamlit run app.py`. Tests run with `pytest`; tests that need the downloaded data or the trained models skip when `data/` or `cache/` is missing.

### Measured LiU-ICE live engine and real-data lab

The **Live engine** replays a healthy LiU-ICE engine-bench trajectory and lets the user inject one of three exact benchmark fault types from the sidebar. The **Real data lab** exposes held-out measured fault runs and shows how one fault propagates across pressure, airflow, fuel, throttle and speed before the consistency guard confirms it. These measured views stay separate from the controlled acoustic simulator in **Sensor fusion**. Prepared data and model bundles are included. Rebuild them from the public source with:

```powershell
python scripts/prepare_liu_guard.py
```

The script downloads the public benchmark only when needed. It validates archive paths, creates five-second features, trains on healthy windows from the training trajectories and saves only the three held-out dashboard runs. It rebuilds the estimators with the installed scikit-learn version rather than loading an incompatible serialized model.

If the bundled files are removed, the dashboard starts in clearly labelled **Demo mode** instead of stopping. Demo mode uses the simulator's attached labels, a synthetic efficiency surface and a synthetic trip replay. Its outputs are illustrative and must not be reported as trained-model results.

## What the models do

- SVM and softmax regression on 44 hand-made features (orders, band energies relative to the noise floor, envelope rhythm, per-cylinder spread, spectral and time statistics, OBD summary values). Softmax weights give the plain-language reasons.
- A small network on the OBD readings alone, to show what OBD can and cannot see.
- A multimodal CNN on sound and vibration spectrograms plus OBD values, one output per fault so two faults can show at once, with an occlusion heat map.
- A healthy-baseline autoencoder with an alarm level that depends on the estimated noise level.
- Valve wear, severity and a 100-to-0 health index with a time-to-service forecast.
- An efficiency model (boosted trees and a small network) trained on Ocslab drivers and checked on held-out drivers and on VED cars, and an optimiser that picks the most efficient gear for the power needed.

## Known limits

- All fault results are on simulated sound and vibration. They show the models find the patterns put into the simulator, not accuracy on a real engine.
- Training noise runs from 10 to 25 dB. At 5 dB the CNN and softmax degrade and the SVM fails; at 0 dB all fail. The autoencoder is reliable down to about 7 to 10 dB.
- Alert latency is 1.0 to 1.7 s against a target of under 1 s.
- The efficiency model validates on held-out Ocslab drivers but not on VED: VED has no torque, so its efficiency rests on an assumed torque. The fuel unit of the Ocslab data is inferred as microlitres per second.
- Fault efficiency losses (misfire 25%, knock 8%, valve wear 3% at full severity), the vehicle parameters, the knock rule and the fuel price (105 rupees per litre) are assumptions; the dashboard sliders change the loss rates.
