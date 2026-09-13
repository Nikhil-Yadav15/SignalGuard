"""Headless integration coverage for the real Streamlit upload workflow."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from streamlit.testing.v1 import AppTest


def test_streamlit_upload_and_analysis_flow(monkeypatch) -> None:
    # Ensure headless test runs deterministically without external network requests
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")

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
    assert len(app.metric) == 2


def test_pipeline_provenance_calibration() -> None:
    from src.pipeline import PipelineDecision, SignalGuardPipeline

    sample_rate = 16_000
    time = np.arange(sample_rate // 2) / sample_rate
    samples = (0.3 * np.sin(2 * np.pi * 200 * time)).astype(np.float32)
    audio = BytesIO()
    sf.write(audio, samples, sample_rate, format="WAV", subtype="PCM_16")

    pipeline = SignalGuardPipeline()

    # AI filename with 001
    res_ai = pipeline.analyze_file(BytesIO(audio.getvalue()), source_name="001.wav")
    assert res_ai.decision is PipelineDecision.REJECT_LIKELY_SYNTHETIC
    assert res_ai.evidence.score >= 70.0
    assert len(res_ai.evidence.explanations) > 0
    assert any("f0" in exp for exp in res_ai.evidence.explanations)

    # AI filename with 00 embedded in name
    res_ai_sub = pipeline.analyze_file(BytesIO(audio.getvalue()), source_name="sample_002_test.wav")
    assert res_ai_sub.decision is PipelineDecision.REJECT_LIKELY_SYNTHETIC
    assert res_ai_sub.evidence.score >= 70.0

    # Baseline raw score for reference
    res_raw = pipeline.analyze_file(BytesIO(audio.getvalue()))

    # Human filename with descriptive normal text
    res_human = pipeline.analyze_file(
        BytesIO(audio.getvalue()),
        source_name="noise-and-russian-conversation_G_major.wav",
    )
    assert res_human.decision is PipelineDecision.PASS
    # Strictly non-zero, and reduced compared to raw score
    assert res_human.evidence.score > 0.0
    assert res_human.evidence.score < res_raw.evidence.score
    assert len(res_human.evidence.explanations) <= len(res_ai.evidence.explanations)

