"""SignalGuard UI package containing styles, presets, sidebar, and tab view renderers."""

from .presets import PRESET_OPTIONS, generate_preset_audio, get_preset_key
from .sidebar import render_sidebar
from .styles import inject_custom_css
from .views import (
    render_decision_banner,
    render_forensics_tab,
    render_gauge_meter,
    render_header,
    render_pipeline_flow_tab,
    render_restoration_tab,
    render_visuals_tab,
)

__all__ = [
    "PRESET_OPTIONS",
    "generate_preset_audio",
    "get_preset_key",
    "inject_custom_css",
    "render_decision_banner",
    "render_forensics_tab",
    "render_gauge_meter",
    "render_header",
    "render_pipeline_flow_tab",
    "render_restoration_tab",
    "render_sidebar",
    "render_visuals_tab",
]
