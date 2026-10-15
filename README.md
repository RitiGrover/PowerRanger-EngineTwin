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
- Valve wear: a tick near 3 kHz at the firing rate.

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
