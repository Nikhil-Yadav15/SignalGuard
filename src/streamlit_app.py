"""Streamlit entry point for local SignalGuard inspection."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

# ``streamlit run src/streamlit_app.py`` executes this file as a script. Add
# the repository root before absolute imports so package imports stay valid.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import soundfile as sf
import streamlit as st

from src.pipeline import SignalGuardPipeline
from src.visualization import (
    plot_before_after,
    plot_spectrogram,
    plot_waveform,
)


def main() -> None:
    st.set_page_config(page_title="SignalGuard", layout="wide")
    st.title("SignalGuard")
    st.caption(
        "Deterministic DSP evidence and targeted audio restoration — "
        "not an authenticity proof."
    )
    upload = st.file_uploader(
        "Audio file",
        type=["wav", "flac", "ogg", "mp3"],
        help="WAV is the most reliable cross-platform test format.",
    )
    if upload is None:
        st.info("Choose an audio file to begin.")
        return

    raw_audio = upload.getvalue()
    if not raw_audio:
        st.error("The selected file is empty.")
        return
    file_suffix = Path(upload.name).suffix.lower().lstrip(".") or "wav"
    mime_type = {
        "wav": "audio/wav",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "mp3": "audio/mpeg",
    }.get(file_suffix, "audio/wav")
    st.audio(raw_audio, format=mime_type)
    st.caption(f"{upload.name} · {len(raw_audio) / 1024:.1f} KiB")
    if not st.button("Analyze audio", type="primary"):
        return

    try:
        with st.spinner("Running deterministic DSP analysis…"):
            result = SignalGuardPipeline().analyze_file(BytesIO(raw_audio))
    except Exception as exc:
        st.error(
            "SignalGuard could not decode or analyze this file. Try a PCM WAV "
            "file; MP3 support depends on locally installed codecs."
        )
        st.exception(exc)
        return

    st.success("Analysis completed.")
    decision_column, score_column, rate_column = st.columns(3)
    decision_column.metric("Decision", result.decision.value)
    score_column.metric("Synthetic Evidence Score", f"{result.evidence.score:.1f}/100")
    rate_column.metric("Analysis sample rate", f"{result.audio.sample_rate:,} Hz")

    st.subheader("Evidence summary")
    st.dataframe(
        [
            {
                "domain": domain.name,
                "score": round(domain.score, 3),
                "triggered rules": ", ".join(domain.triggered_rules) or "none",
            }
            for domain in result.evidence.domain_scores
        ],
        width="stretch",
    )
    if result.evidence.explanations:
        with st.expander("Rule explanations"):
            for explanation in result.evidence.explanations:
                st.write(f"- {explanation}")

    waveform_column, spectrogram_column = st.columns(2)
    with waveform_column:
        st.pyplot(
            plot_waveform(
                result.audio.samples,
                result.audio.sample_rate,
                title="Canonical input",
            )
        )
    with spectrogram_column:
        st.pyplot(
            plot_spectrogram(
                result.audio.samples,
                result.audio.sample_rate,
            )
        )

    st.subheader("Corruption indicators")
    st.dataframe(
        result.corruption_before.to_dict()["detections"],
        width="stretch",
    )

    if result.restored_samples is not None:
        restored = np.asarray(result.restored_samples, dtype=np.float32)
        st.pyplot(
            plot_before_after(
                result.audio.samples,
                restored,
                result.audio.sample_rate,
            )
        )
        buffer = BytesIO()
        sf.write(
            buffer,
            restored,
            result.audio.sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
        st.download_button(
            "Download restored WAV",
            buffer.getvalue(),
            "signalguard_restored.wav",
            "audio/wav",
        )

    if result.quality is not None:
        st.subheader("Restoration quality checks")
        st.dataframe(
            result.quality.to_dict()["checks"],
            width="stretch",
        )


if __name__ == "__main__":
    main()
