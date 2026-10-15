import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "cache" / "model_cnn.pkl").exists(), reason="trained models are not in cache/")
TIMEOUT = 180


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()


def chart_titles(at):
    return [json.loads(el.proto.spec)["layout"]["title"]["text"] for el in at.get("plotly_chart")]


def test_the_dashboard_starts_without_errors_and_has_all_its_sections(app):
    assert not app.exception
    assert [t.label for t in app.tabs] == [
        "Live engine", "Signatures", "Diagnosis", "Efficiency and sweet spot", "Health", "Data explorer", "Real recording", "Results",
    ]


def test_a_healthy_engine_shows_a_healthy_status(app):
    assert any(s.value.startswith("Healthy") for s in app.success)


def test_every_chart_is_tagged_real_simulated_or_mixed(app):
    titles = chart_titles(app)

    assert len(titles) >= 16  # 5 gauges and chunk, 5 signatures, probabilities, 2 efficiency, health, confusion
    assert all(t.endswith(("[Real]", "[Simulated]", "[Mixed]")) for t in titles), [t for t in titles if not t.endswith(("[Real]", "[Simulated]", "[Mixed]"))]


def test_injecting_a_severe_misfire_turns_the_status_to_fault_and_names_it():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()
    at.selectbox(key="fault").select("misfire").run()  # the severity slider is disabled until a fault is chosen
    at.slider(key="severity").set_value(1.0).run()

    assert not at.exception
    assert any("Fault: misfire" in e.value for e in at.error)
    assert any("loses" in w.value for w in at.warning)  # the efficiency lost to the fault is shown


def test_the_controls_for_drive_mode_noise_loss_assumptions_and_trip_replay_are_present(app):
    keys = {"drive_mode", "fault", "severity", "snr", "loss_misfire", "loss_knock", "loss_valve", "trip", "position"}
    found = {w.key for w in list(app.sidebar.selectbox) + list(app.sidebar.slider) + list(app.sidebar.select_slider)}

    assert keys <= found
