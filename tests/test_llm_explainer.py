"""Unit tests for the LLM acoustic forensics explainer module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.llm import (
    ForensicSummaryResult,
    generate_forensic_summary,
    resolve_llm_credentials,
    resolve_openrouter_credentials,
)
from src.llm.explainer import _clean_json_text


def test_clean_json_text() -> None:
    fenced = '```json\n{"headline": "Test Title", "overview": "Test Overview"}\n```'
    assert _clean_json_text(fenced) == '{"headline": "Test Title", "overview": "Test Overview"}'

    plain = '{"headline": "Plain"}'
    assert _clean_json_text(plain) == '{"headline": "Plain"}'


def test_resolve_gemini_credentials_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-gemini-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    provider, api_key, model = resolve_llm_credentials()
    assert provider == "gemini"
    assert api_key == "AIzaSy-gemini-test-key"
    assert model == "gemini-2.5-flash"


def test_resolve_google_api_key_alias(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSy-google-test-key")
    monkeypatch.setenv("GOOGLE_MODEL", "gemini-1.5-flash")

    provider, api_key, model = resolve_llm_credentials()
    assert provider == "gemini"
    assert api_key == "AIzaSy-google-test-key"
    assert model == "gemini-1.5-flash"


def test_resolve_openrouter_fallback_from_env(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    provider, api_key, model = resolve_llm_credentials()
    assert provider == "openrouter"
    assert api_key == "sk-or-test-key"
    assert model == "openai/gpt-4o-mini"

    # Backward compatibility
    legacy_key, legacy_model = resolve_openrouter_credentials()
    assert legacy_key == "sk-or-test-key"
    assert legacy_model == "openai/gpt-4o-mini"


def test_resolve_credentials_none(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with patch("src.llm.explainer.resolve_llm_credentials", return_value=("none", None, "gemini-2.5-flash")):
        provider, api_key, model = resolve_llm_credentials()
        assert provider == "none"
        assert api_key is None


def test_generate_forensic_summary_fallback_without_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with patch("src.llm.explainer.resolve_llm_credentials", return_value=("none", None, "gemini-2.5-flash")):
        res = generate_forensic_summary(
            score=82.4,
            decision_text="REJECT_LIKELY_SYNTHETIC",
            explanations=(
                "f0: Monotone pitch contour with variance below human range",
                "breath: Continuous speech burst exceeds natural human breathing capacity",
            ),
        )

        assert isinstance(res, ForensicSummaryResult)
        assert res.is_fallback is True
        assert len(res.standout_facts) == 2
        assert "Monotone pitch" in res.standout_facts[0]
        assert res.auditory_cues != ""
        assert res.error_message is not None


def test_generate_forensic_summary_clean_baseline(monkeypatch) -> None:
    with patch("src.llm.explainer.resolve_llm_credentials", return_value=("none", None, "gemini-2.5-flash")):
        res = generate_forensic_summary(
            score=12.0,
            decision_text="PASS",
            explanations=(),
        )

        assert isinstance(res, ForensicSummaryResult)
        assert res.is_fallback is True
        assert "Natural Human Speech" in res.headline
        assert len(res.standout_facts) > 0


def test_generate_forensic_summary_with_mock_gemini() -> None:
    mock_response_json = """
    ```json
    {
        "headline": "Robotic Speech with Unnatural Continuity",
        "overview": "The voice displays hallmark indicators of AI synthesis, particularly lacking human breathing cadence and natural pitch inflection.",
        "standout_facts": [
            "Pitch is unnaturally flat without natural human inflection.",
            "Speech ran for 6 continuous seconds with zero breath pauses."
        ],
        "auditory_cues": "Listen for robotic stability in pitch and absent breathing sounds between clauses."
    }
    ```
    """

    mock_llm_instance = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = mock_response_json
    mock_llm_instance.stream.return_value = [mock_chunk]
    mock_llm_instance.invoke.return_value = mock_chunk

    with (
        patch("src.llm.explainer.resolve_llm_credentials", return_value=("gemini", "AIzaSy-mock-key", "gemini-2.5-flash")),
        patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm_instance),
    ):
        res = generate_forensic_summary(
            score=88.5,
            decision_text="REJECT_LIKELY_SYNTHETIC",
            explanations=("f0: Pitch flat", "breath: No breath"),
        )

        assert res.is_fallback is False
        assert res.provider == "gemini"
        assert res.headline == "Robotic Speech with Unnatural Continuity"
        assert len(res.standout_facts) == 2
        assert "zero breath pauses" in res.standout_facts[1]
        assert "Listen for robotic stability" in res.auditory_cues


def test_generate_forensic_summary_gemini_exception_fallback() -> None:
    mock_llm_instance = MagicMock()
    mock_llm_instance.stream.side_effect = RuntimeError("Gemini API quota exceeded")
    mock_llm_instance.invoke.side_effect = RuntimeError("Gemini API quota exceeded")

    with (
        patch("src.llm.explainer.resolve_llm_credentials", return_value=("gemini", "AIzaSy-mock-key", "gemini-2.5-flash")),
        patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm_instance),
    ):
        res = generate_forensic_summary(
            score=88.5,
            decision_text="REJECT_LIKELY_SYNTHETIC",
            explanations=("f0: Pitch flat",),
        )

        assert res.is_fallback is True
        assert res.error_message is not None
        assert "RuntimeError" in res.error_message
