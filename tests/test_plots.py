import pandas as pd

from enginetwin.plots import efficiency_heatmap


def cells():
    return pd.DataFrame(
        {
            "rpm": [1000.0, 1000.0, 2000.0, 2000.0],
            "torque": [10.0, 30.0, 10.0, 30.0],
            "efficiency": [0.10, 0.15, 0.17, 0.25],
            "samples": [50, 60, 70, 80],
        }
    )


def test_heatmap_shows_every_cell_and_marks_the_sweet_spot():
    fig = efficiency_heatmap(cells(), source="real")

    heat, marker = fig.data
    assert len(heat.x) * len(heat.y) == 4
    assert (marker.x[0], marker.y[0]) == (2000.0, 30.0)  # the 0.25 cell


def test_heatmap_labels_axes_with_units_and_tags_the_data_source():
    fig = efficiency_heatmap(cells(), source="real")

    assert "rpm" in fig.layout.xaxis.title.text
    assert "Nm" in fig.layout.yaxis.title.text
    assert "Real" in fig.layout.title.text
