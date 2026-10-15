import numpy as np
import pytest

from enginetwin.schema import Window


def make_window(**overrides):
    fields = dict(
        sound=np.zeros(2 * 16000),
        sound_fs=16000,
        vibration=np.zeros(2 * 8000),
        vibration_fs=8000,
        obd={"rpm": 900.0, "speed": 0.0},
        fault="healthy",
        severity=0.0,
        source="simulated",
    )
    fields.update(overrides)
    return Window(**fields)


def test_window_reports_duration_in_seconds():
    w = make_window()
    assert w.duration_s == 2.0


def test_window_rejects_unknown_fault_label():
    with pytest.raises(ValueError, match="fault"):
        make_window(fault="exploded")


def test_window_rejects_unknown_source_tag():
    with pytest.raises(ValueError, match="source"):
        make_window(source="made-up")


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_window_rejects_severity_outside_zero_to_one(bad):
    with pytest.raises(ValueError, match="severity"):
        make_window(fault="misfire", severity=bad)
