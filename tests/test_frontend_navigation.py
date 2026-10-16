"""Frontend smoke checks with the supplied artifacts (no synthetic training required)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_all_sections_render_and_results_use_available_artifacts():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not app.exception
    assert not any("01 / Overview" in element.value for element in app.markdown)
    app.button(key="enter_engine_twin").click().run()
    assert not app.exception
    navigation = app.radio(key="workspace_section")
    assert len(navigation.options) == 11
    assert "Engine map" in navigation.options
    assert "Sweet spot" in navigation.options
    app.toggle(key="guard_enabled").set_value(False).run()
    app.slider(key="manual_bias_engine_speed").set_value(10).run()
    for section in navigation.options:
        app.radio(key="workspace_section").set_value(section).run()
        assert not app.exception, (section, [error.message for error in app.exception])
        if section == "Results":
            assert len(app.dataframe) >= 3
            assert app.metric[0].value == "3"
            assert not any("unavailable in demo mode" in info.value for info in app.info)
    app.radio(key="workspace_section").set_value("Live engine").run()
    assert not app.exception
    assert app.toggle(key="guard_enabled").value is False
    assert app.slider(key="manual_bias_engine_speed").value == 10


def test_live_chart_has_fixed_scale_and_rolling_time_window():
    import numpy as np
    import pandas as pd
    from simulation_dashboard import _timeline, VISIBLE_WINDOWS

    rows = 50
    frame = pd.DataFrame({
        "window_end_s": np.arange(1, rows + 1) * 5.,
        "injection_time_s": np.nan,
        **{f"{sensor}__mean": np.arange(rows, dtype=float)
           for sensor in ("engine_speed", "intake_manifold_pressure", "air_mass_flow", "throttle_position")},
    })
    evidence = pd.DataFrame({"f_pim": np.zeros(rows)})
    confirmed = evidence.astype(bool)
    first = _timeline(frame, evidence, confirmed, 35, True, float("inf"))
    second = _timeline(frame, evidence, confirmed, 36, True, float("inf"))
    assert len(second.data[0].x) == VISIBLE_WINDOWS
    assert first.layout.yaxis.range == second.layout.yaxis.range
    assert second.layout.xaxis.range[1] - first.layout.xaxis.range[1] == 5
    np.testing.assert_allclose(first.data[0].y[1:], second.data[0].y[:-1])
