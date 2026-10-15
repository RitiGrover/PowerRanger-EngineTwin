import pytest

from enginetwin.losses import DEFAULT_LOSS_PER_SEVERITY, degraded_efficiency, efficiency_loss


def test_the_default_assumed_loss_rates_are_the_stated_ones():
    assert DEFAULT_LOSS_PER_SEVERITY == {"misfire": 0.25, "knock": 0.08, "valve_wear": 0.03}


@pytest.mark.parametrize(
    "fault, severity, expected",
    [
        ("healthy", 0.0, 0.0),
        ("misfire", 0.5, 0.125),  # 0.25 x 0.5
        ("knock", 1.0, 0.08),
        ("valve_wear", 0.4, 0.012),  # 0.03 x 0.4
        ("misfire+knock", 1.0, 0.33),  # 0.25 + 0.08
        ("knock+valve_wear", 0.5, 0.055),  # (0.08 + 0.03) x 0.5
    ],
)
def test_the_loss_is_the_rate_times_severity_added_over_the_faults_present(fault, severity, expected):
    assert efficiency_loss(fault, severity) == pytest.approx(expected)


def test_loss_rates_can_be_overridden_for_the_sliders():
    assert efficiency_loss("misfire", 0.5, rates={"misfire": 0.4}) == pytest.approx(0.2)
    assert efficiency_loss("misfire+knock", 1.0, rates={"misfire": 0.4}) == pytest.approx(0.48)  # knock keeps its default


def test_degraded_efficiency_is_the_healthy_value_less_the_loss_fraction():
    assert degraded_efficiency(0.30, "misfire", 0.5) == pytest.approx(0.30 * (1 - 0.125))
    assert degraded_efficiency(0.30, "healthy", 0.0) == pytest.approx(0.30)
