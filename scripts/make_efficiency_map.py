"""Build the real-data efficiency map from the Ocslab file and write it to cache/efficiency_map.html."""
from pathlib import Path

from enginetwin.efficiency_map import efficiency_map
from enginetwin.ocslab import load_ocslab
from enginetwin.plots import efficiency_heatmap

root = Path(__file__).resolve().parents[1]
csv = next((root / "data" / "osclab").glob("*.csv"))
out = root / "cache" / "efficiency_map.html"

fig = efficiency_heatmap(efficiency_map(load_ocslab(csv)), source="real")
out.parent.mkdir(exist_ok=True)
fig.write_html(out, include_plotlyjs="cdn")
print(out)
