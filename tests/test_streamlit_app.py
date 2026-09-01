"""Headless integration coverage for the real Streamlit upload workflow."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from streamlit.testing.v1 import AppTest


def test_streamlit_upload_and_analysis_flow() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate // 2) / sample_rate
    samples = (0.3 * np.sin(2 * np.pi * 200 * time)).astype(np.float32)
    audio = BytesIO()
    sf.write(audio, samples, sample_rate, format="WAV", subtype="PCM_16")
    app_path = Path(__file__).resolve().parents[1] / "src" / "streamlit_app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=20)
    assert not app.exception
    app.get("file_uploader")[0].upload(
        "generated_tone.wav",
        audio.getvalue(),
        "audio/wav",
    ).run(timeout=20)
    assert len(app.button) == 1

    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert [message.value for message in app.success] == ["Analysis completed."]
    assert len(app.metric) == 3
