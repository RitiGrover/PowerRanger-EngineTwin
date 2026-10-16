"""Regression coverage for stale result-class identities after UI hot reloads."""
import pickle
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from enginetwin import diagnosis
from enginetwin.demo import DemoDiagnoser
from enginetwin.schema import FAULTS
from enginetwin.simulator import simulate_window

ROOT = Path(__file__).resolve().parents[1]


def test_fault_selection_survives_a_reloaded_result_class(monkeypatch):
    # A cached diagnoser can retain the old class while Streamlit reloads its
    # module. Reproduce the exact pickle identity mismatch in the user's report.
    previous_result = DemoDiagnoser().diagnose(simulate_window(rpm=2400, fault="misfire", severity=.8))
    replacement = type("DiagnosisResult", (diagnosis.DiagnosisResult,),
                       {"__module__": diagnosis.__name__})
    monkeypatch.setattr(diagnosis, "DiagnosisResult", replacement)
    with pytest.raises(pickle.PicklingError, match="not the same object"):
        pickle.dumps(previous_result)

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    app.session_state["entered_engine_twin"] = True
    app.session_state["workspace_section"] = "Sensor fusion"
    app.run()
    assert not app.exception
    for fault in FAULTS:
        app.selectbox(key="fault").set_value(fault).run()
        assert not app.exception, (fault, [e.message for e in app.exception])
        if fault == "healthy":
            assert any("Healthy" in item.value for item in app.success)
        else:
            assert any(fault.replace("+", " and ").replace("_", " ") in item.value
                       for item in app.error)
    # Exercise a cache hit after returning to an already selected fault.
    app.selectbox(key="fault").set_value("misfire").run()
    assert not app.exception
