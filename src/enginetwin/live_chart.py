"""Persistent frontend Plotly surface for the periodically refreshed replay."""
from functools import lru_cache

import streamlit as st
from plotly.offline import get_plotlyjs


@lru_cache(maxsize=1)
def _component():
    # Ship the installed Plotly runtime locally; playback needs no CDN connection.
    return st.components.v2.component(
        "enginetwin_live_chart",
        html='<div class="et-live-plot" role="img" aria-label="Live engine chart"></div>',
        js=get_plotlyjs() + """
export default function({parentElement, data}) {
    const plot = parentElement.querySelector('.et-live-plot');
    plot.setAttribute('aria-label', data.label);
    const figure = JSON.parse(data.figure);
    plot.style.height = `${figure.layout.height}px`;
    if (figure.layout.height > 250) {
        figure.layout.margin.t = plot.clientWidth < 480 ? 150 : 85;
        figure.layout.legend.y = 1.02;
        figure.layout.legend.yanchor = 'bottom';
    }
    // Keep the graph DOM and legend state while new samples and axis ranges arrive.
    window.Plotly.react(plot, figure.data, figure.layout, {
        displayModeBar: false, responsive: true, scrollZoom: false
    });
    const observer = new ResizeObserver(() => {
        if (plot.isConnected && plot._fullLayout) window.Plotly.Plots.resize(plot);
    });
    observer.observe(plot);
    return () => observer.disconnect();
}
""",
        isolate_styles=False,
    )


def live_chart(figure, *, key: str, label: str) -> None:
    figure.update_layout(uirevision=key, transition_duration=0)
    _component()(key=key, data={"figure": figure.to_json(), "label": label},
                 height=int(figure.layout.height), width="stretch")
