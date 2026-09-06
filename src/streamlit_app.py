"""Streamlit entry point for local SignalGuard inspection.

Deterministic audio forensics and restoration dashboard based on dsp_project_prompt.md.
"""

from __future__ import annotations

import contextlib
import sys
from io import BytesIO
from pathlib import Path

with contextlib.suppress(ImportError):
    import numba  # noqa: F401 - eagerly initialize to prevent lazy_loader circular imports

# ``streamlit run src/streamlit_app.py`` executes this file as a script. Add
# the repository root before absolute imports so package imports stay valid.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.pipeline import SignalGuardPipeline
from src.ui import (
    PRESET_OPTIONS,
    generate_preset_audio,
    get_preset_key,
    inject_custom_css,
    render_decision_banner,
    render_forensics_tab,
    render_gauge_meter,
    render_header,
    render_pipeline_flow_tab,
    render_restoration_tab,
    render_sidebar,
    render_visuals_tab,
)


def main() -> None:
    st.set_page_config(
        page_title="SignalGuard · Audio Forensics & Restoration",
        page_icon="🛡️",
        layout="wide",
    )
    inject_custom_css()
    render_sidebar()
    render_header()

    # Audio input section: File upload or benchmark preset
    st.markdown("### 1. Select Audio Input")
    col_upload, col_preset = st.columns([3, 2])

    with col_upload:
        upload = st.file_uploader(
            "Upload Audio File (.wav, .flac, .ogg, .mp3)",
            type=["wav", "flac", "ogg", "mp3"],
            help="WAV is the most reliable cross-platform test format.",
        )

    with col_preset:
        preset_choice = st.selectbox(
            "Or load a built-in benchmark preset:",
            options=PRESET_OPTIONS,
            help="Synthesized test audio from dsp_project_prompt.md test suites.",
        )

    raw_audio: bytes | None = None
    source_name: str = ""
    file_suffix: str = "wav"

    if upload is not None:
        raw_audio = upload.getvalue()
        source_name = upload.name
        file_suffix = Path(upload.name).suffix.lower().lstrip(".") or "wav"
    elif preset_choice != "None (Use uploaded file)":
        preset_key = get_preset_key(preset_choice)
        raw_audio, source_name = generate_preset_audio(preset_key)
        file_suffix = "wav"

    if not raw_audio:
        st.info("Choose an audio file or select a preset to begin analysis.")
        return

    mime_type = {
        "wav": "audio/wav",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "mp3": "audio/mpeg",
    }.get(file_suffix, "audio/wav")

    # Audio playback and metadata
    st.markdown("#### Audio Preview")
    audio_col, meta_col = st.columns([3, 2])
    with audio_col:
        st.audio(raw_audio, format=mime_type)
    with meta_col:
        st.caption(f"📁 **Source**: `{source_name}`")
        st.caption(f"📦 **Size**: `{len(raw_audio) / 1024:.1f} KiB` · **Format**: `{file_suffix.upper()}`")

    # Primary analysis button (single button required by integration tests)
    if not st.button("Analyze audio", type="primary"):
        return

    try:
        with st.spinner("Running deterministic DSP forensic & quality pipeline…"):
            result = SignalGuardPipeline().analyze_file(BytesIO(raw_audio))
    except Exception as exc:  # noqa: BLE001 - user audio may raise any decode error
        st.error(
            "SignalGuard could not decode or analyze this file. Try a standard PCM WAV "
            "file; compressed formats depend on local libsndfile codecs."
        )
        st.exception(exc)
        return

    # Success notification required by test suite
    st.success("Analysis completed.")

    # Top 3 metrics (Strictly 3 st.metric calls to preserve test_streamlit_app.py contract)
    decision_column, score_column, rate_column = st.columns(3)
    decision_column.metric("Decision", result.decision.value)
    score_column.metric("Synthetic Evidence Score", f"{result.evidence.score:.1f}/100")
    rate_column.metric("Analysis sample rate", f"{result.audio.sample_rate:,} Hz")

    # Executive Decision Banner & Gauge
    render_decision_banner(result)
    render_gauge_meter(result.evidence.score)

    # Interactive Inspection Tabs
    tab_forensics, tab_restoration, tab_visuals, tab_flow = st.tabs([
        "🔬 Forensics & Deepfake Detection",
        "🛠️ Signal Quality & Restoration",
        "📊 Visual Waveforms & Spectrograms",
        "🧭 Pipeline Decision Flow",
    ])

    with tab_forensics:
        render_forensics_tab(result)

    with tab_restoration:
        render_restoration_tab(result)

    with tab_visuals:
        render_visuals_tab(result)

    with tab_flow:
        render_pipeline_flow_tab(result)


if __name__ == "__main__":
    main()
