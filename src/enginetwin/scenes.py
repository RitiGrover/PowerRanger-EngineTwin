"""HTML for the two 3D scenes (landing page and engine map). Presentation only.

The pages are self-contained documents rendered inside Streamlit iframes. Three.js comes from a pinned CDN
build; the engine itself is built procedurally in web/engine3d.js, so there is no model file to download."""
import json
from pathlib import Path

from enginetwin.theme import logo_mark

WEB = Path(__file__).parent / "web"


def _read(name: str) -> str:
    return (WEB / name).read_text(encoding="utf8")


def _inline(template: str) -> str:
    return template.replace("/*__ENGINE3D__*/", _read("engine3d.js"))


def landing_html() -> str:
    return _inline(_read("landing.html")).replace("__LOGO__", logo_mark(34))


def engine_map_html(payload: dict) -> str:
    # "</" inside a JSON <script> body could close the tag early, so it is escaped.
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return _inline(_read("engine_map.html")).replace("__DATA__", data)


def engine_viewer_html(label: str, severity: float, rpm: float, load: float) -> str:
    """Compact rotating engine for the panels that used to show a cartoon engine."""
    nominal = label == "healthy"
    health = max(18, round(100 - severity * (34 if not nominal else 0)))
    data = json.dumps({"label": label, "severity": float(severity), "rpm": float(rpm), "load": float(load), "health": int(health)})
    return _inline(_read("viewer.html")).replace("__DATA__", data.replace("</", "<\\/"))
