import numpy as np
import pandas as pd
import pytest

from enginetwin import dashboard as d


def tag_of(fig):
    return fig.layout.title.text


@pytest.mark.parametrize("source, shown", [("real", "Real"), ("simulated", "Simulated"), ("mixed", "Mixed")])
def test_every_title_carries_its_data_tag(source, shown):
    assert d.tagged_title("Spectrum", source) == f"Spectrum [{shown}]"


def test_an_unknown_tag_is_refused():
    with pytest.raises(ValueError):
        d.tagged_title("Spectrum", "measured")


def test_each_chart_builder_tags_its_figure():
    spec = np.random.default_rng(0).normal(size=(64, 64))
    cells = pd.DataFrame({"rpm": [1000.0, 1000.0, 2000.0, 2000.0], "torque": [10.0, 30.0, 10.0, 30.0], "efficiency": [0.1, 0.15, 0.17, 0.25], "samples": [50] * 4})
    replay = pd.DataFrame({c: np.arange(5.0) for c in ["time_s", "speed_kmh", "actual_gear", "opt_gear", "actual_rpm", "opt_rpm", "actual_eff", "opt_eff", "actual_fuel_l_h", "opt_fuel_l_h"]})
    figures = {
        "gauge": (d.gauge_figure("Engine speed", 2400, 0, 6000, "rpm", "simulated"), "Simulated"),
        "spectrogram": (d.spectrogram_figure(spec, 2.0, "Sound", "simulated"), "Simulated"),
        "orders": (d.order_spectrum_figure(np.linspace(0, 8, 50), np.ones(50), "simulated"), "Simulated"),
        "envelope": (d.envelope_figure(np.linspace(0, 300, 50), np.ones(50), 2400, "simulated"), "Simulated"),
        "cylinders": (d.cylinder_figure({1: 1.0, 2: 0.1, 3: 1.0, 4: 1.0}, "simulated"), "Simulated"),
        "probabilities": (d.probability_figure({"misfire": 0.9, "knock": 0.1, "valve_wear": 0.2}, "simulated"), "Simulated"),
        "heat": (d.heat_overlay_figure(spec, np.abs(spec), "simulated"), "Simulated"),
        "efficiency": (d.efficiency_map_figure(cells, (1000.0, 10.0), (2000.0, 30.0), "real"), "Real"),
        "series": (d.series_figure(replay, "mixed"), "Mixed"),
        "health": (d.health_figure(np.arange(5), np.linspace(100, 20, 5), np.linspace(100, 22, 5), 7.0, "simulated"), "Simulated"),
        "confusion": (d.confusion_figure(np.eye(3, dtype=int), ["a", "b", "c"], "Confusion", "simulated"), "Simulated"),
    }
    for name, (fig, shown) in figures.items():
        assert f"[{shown}]" in tag_of(fig), name


def test_the_order_spectrum_marks_the_key_orders_with_their_names():
    fig = d.order_spectrum_figure(np.linspace(0, 8, 50), np.ones(50), "simulated")
    notes = " ".join(a.text for a in fig.layout.annotations)

    for expected in ("0.5 (misfire)", "1", "2 (firing)", "4"):
        assert expected in notes


def test_the_envelope_chart_marks_the_valve_and_firing_rates_for_the_given_speed():
    fig = d.envelope_figure(np.linspace(0, 300, 50), np.ones(50), 2400, "simulated")
    xs = sorted(round(s.x0, 1) for s in fig.layout.shapes)

    assert xs == [20.0, 80.0]  # 2400 / 120 once per cycle, 2400 / 30 once per firing


def test_the_weakest_cylinder_gets_its_own_colour():
    bar = d.cylinder_figure({1: 1.0, 2: 0.1, 3: 1.0, 4: 1.0}, "simulated").data[0]

    assert bar.marker.color[1] != bar.marker.color[0] and len(set(bar.marker.color)) == 2


def test_the_efficiency_map_marks_the_current_and_the_optimal_point():
    cells = pd.DataFrame({"rpm": [1000.0, 2000.0], "torque": [10.0, 30.0], "efficiency": [0.1, 0.25], "samples": [50, 50]})
    fig = d.efficiency_map_figure(cells, (1000.0, 10.0), (2000.0, 30.0), "real")
    named = {t.name: (t.x[0], t.y[0]) for t in fig.data if t.name}

    assert named["Current point"] == (1000.0, 10.0) and named["Optimal point"] == (2000.0, 30.0)


def test_the_series_chart_overlays_actual_and_optimised_for_efficiency_fuel_and_rpm():
    replay = pd.DataFrame({c: np.arange(5.0) for c in ["time_s", "speed_kmh", "actual_gear", "opt_gear", "actual_rpm", "opt_rpm", "actual_eff", "opt_eff", "actual_fuel_l_h", "opt_fuel_l_h"]})
    names = [t.name for t in d.series_figure(replay, "mixed").data]

    assert names.count("Actual") == 3 and names.count("Optimised") == 3


def test_drive_modes_run_the_engine_progressively_harder():
    eco, normal, sport = (d.drive_mode_point(m) for m in ("Eco", "Normal", "Sport"))

    assert eco["rpm"] < normal["rpm"] < sport["rpm"] and eco["load"] < normal["load"] < sport["load"]
    with pytest.raises(ValueError):
        d.drive_mode_point("Race")


def test_the_data_explorer_says_which_data_trained_which_model_and_which_is_held_out_for_the_demo():
    table = d.data_provenance()

    assert list(table.columns) == ["data", "kind", "trains", "demonstrates", "notes"]
    by = table.set_index("data")
    assert "efficiency model" in by.loc["Ocslab drivers A, B, D, E, F, G, I, J", "trains"]
    assert "efficiency model" not in by.loc["Ocslab drivers C, H", "trains"] and by.loc["Ocslab drivers C, H", "demonstrates"] != ""
    assert by.loc["VED (ICE cars)", "trains"] == "" and by.loc["Simulated sound and vibration", "kind"] == "Simulated"
    assert set(table.kind) <= {"Real", "Simulated", "Mixed"}
