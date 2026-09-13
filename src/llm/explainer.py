"""Acoustic forensics explainer powered by LangChain and Google Gemini.

Translates deterministic DSP findings into plain-language summaries with standout
facts and auditory cues for non-technical users.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

# Load .env at import time
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"


@dataclass(frozen=True)
class ForensicSummaryResult:
    """Structured human-understandable explanation of forensic findings."""

    headline: str
    overview: str
    standout_facts: tuple[str, ...] = field(default_factory=tuple)
    auditory_cues: str = ""
    is_fallback: bool = False
    error_message: str | None = None
    provider: str = "none"


def resolve_llm_credentials() -> tuple[str, str | None, str]:
    """Retrieve configured LLM provider, API key, and model name.

    Checks Google Gemini first (GEMINI_API_KEY / GOOGLE_API_KEY),
    then falls back to OpenRouter (OPENROUTER_API_KEY).
    Supports Streamlit secrets (st.secrets) and environment variables (.env / OS).

    Returns:
        tuple[provider, api_key, model_name] where provider is 'gemini', 'openrouter', or 'none'.
    """
    gemini_key: str | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    openrouter_key: str | None = None
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL

    # 1. Check Streamlit secrets
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            # Check Gemini in st.secrets
            for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                if k in st.secrets:
                    val = str(st.secrets[k]).strip()
                    if val:
                        gemini_key = val
                        break
            if not gemini_key and "gemini" in st.secrets and isinstance(st.secrets["gemini"], dict):
                val = str(st.secrets["gemini"].get("api_key", "")).strip()
                if val:
                    gemini_key = val

            for mk in ("GEMINI_MODEL", "GOOGLE_MODEL"):
                if mk in st.secrets:
                    m = str(st.secrets[mk]).strip()
                    if m:
                        gemini_model = m
                        break

            # Check OpenRouter in st.secrets
            if "OPENROUTER_API_KEY" in st.secrets:
                val = str(st.secrets["OPENROUTER_API_KEY"]).strip()
                if val:
                    openrouter_key = val
            elif "openrouter" in st.secrets and isinstance(st.secrets["openrouter"], dict):
                val = str(st.secrets["openrouter"].get("api_key", "")).strip()
                if val:
                    openrouter_key = val

            if "OPENROUTER_MODEL" in st.secrets:
                m = str(st.secrets["OPENROUTER_MODEL"]).strip()
                if m:
                    openrouter_model = m
    except Exception:
        pass

    # 2. Check environment variables (.env or OS)
    if not gemini_key:
        for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            val = os.getenv(k)
            if val and val.strip():
                gemini_key = val.strip()
                break

    for mk in ("GEMINI_MODEL", "GOOGLE_MODEL"):
        val = os.getenv(mk)
        if val and val.strip():
            gemini_model = val.strip()
            break

    if not openrouter_key:
        val = os.getenv("OPENROUTER_API_KEY")
        if val and val.strip():
            openrouter_key = val.strip()

    val = os.getenv("OPENROUTER_MODEL")
    if val and val.strip():
        openrouter_model = val.strip()

    # Prioritize Gemini
    if gemini_key:
        return "gemini", gemini_key, gemini_model
    if openrouter_key:
        return "openrouter", openrouter_key, openrouter_model

    return "none", None, gemini_model


def resolve_openrouter_credentials() -> tuple[str | None, str]:
    """Backward compatibility helper returning (api_key, model_name)."""
    provider, api_key, model = resolve_llm_credentials()
    return api_key, model


def _clean_json_text(text: str) -> str:
    """Strip markdown code fences and extraneous text from JSON response."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def _build_fallback_summary(
    score: float,
    decision_text: str,
    explanations: tuple[str, ...],
    error_message: str | None = None,
) -> ForensicSummaryResult:
    """Generate a clean rule-based summary when LLM is unavailable or unconfigured."""
    if not explanations:
        return ForensicSummaryResult(
            headline="Natural Human Speech Characteristics",
            overview=(
                f"The audio exhibits natural human acoustic traits with an evidence score of {score:.1f}%. "
                "No synthetic vocoder anomalies or severe acoustic thresholds were crossed."
            ),
            standout_facts=(
                "Pitch and fundamental frequency show organic human modulation.",
                "Glottal excitation pulses match biological vocal tract dynamics.",
                "Speech respiration and pause cadence follow natural human rhythm.",
            ),
            auditory_cues="The voice sounds natural, with realistic breathing pauses and expressive pitch inflection.",
            is_fallback=True,
            error_message=error_message,
            provider="heuristic",
        )

    facts: list[str] = []
    for exp in explanations[:4]:
        if ": " in exp:
            domain, detail = exp.split(": ", 1)
            facts.append(f"{domain.upper()}: {detail}")
        else:
            facts.append(exp)

    headline = "Acoustic Discrepancies Flagged" if score < 70.0 else "Likely Synthetic Voice Detected"
    overview = (
        f"Forensic analysis flagged {len(explanations)} acoustic metric(s) exceeding normal human baselines "
        f"(Synthetic Evidence Score: {score:.1f}%). {decision_text}."
    )

    return ForensicSummaryResult(
        headline=headline,
        overview=overview,
        standout_facts=tuple(facts),
        auditory_cues=(
            "Listen for flat robotic inflection, unnatural continuous speech without breathing, "
            "or abrupt step-function audio cutoffs at phrase ends."
        ),
        is_fallback=True,
        error_message=error_message,
        provider="heuristic",
    )


def _get_prompts(
    score: float,
    decision_text: str,
    explanations: tuple[str, ...],
    domain_summary: str,
) -> tuple[str, str]:
    """Generate system and user prompts for the forensic summary."""
    system_prompt = (
        "You are an expert audio forensic scientist and speech signal processing analyst. "
        "Your role is to analyze acoustic detector findings from an automated forensics pipeline "
        "and synthesize a clear, informative summary using simple language that anyone can understand.\n\n"
        "Guidelines:\n"
        "1. Use simple language: Write clearly and concisely in plain English. Avoid unnecessary academic jargon or overly dense phrasing so that a non-technical reader can easily grasp the findings.\n"
        "2. Blend simple explanations with exact physical acoustic details: For the findings that stood out the most, explicitly cite the relevant technical metric names or physical mechanisms "
        "(e.g., 'mean_zero_crossing_rate crossed its configured spectral threshold', "
        "'Glottal residual excess kurtosis is abnormally low (smeared vocoder pulses)', "
        "'F0 pitch variation std below human threshold (monotone contour)', "
        "'Speech burst exceeds natural human respiratory pause limits'). "
        "State the technical metric, but explain what it physically means in simple, accessible terms (e.g., natural vocal cord vibration, human breathing limits, or synthetic neural voice generator artifacts).\n"
        "3. Provide 2 to 4 key observations that stood out the most.\n"
        "4. Explain specific auditory cues for human verification in playback (e.g., missing breathing pauses, unnatural pitch stability, abrupt word truncations) in simple terms.\n"
        "5. Respond strictly in valid JSON format with the following schema:\n"
        "{\n"
        '  "headline": "A concise, clear 4-8 word assessment title in simple language",\n'
        '  "overview": "2-3 simple, clear sentences summarizing whether the voice exhibits authentic human speech or synthetic vocoder artifacts, and the primary reason why",\n'
        '  "standout_facts": [\n'
        '    "First key finding citing the technical metric name and explaining what it means in simple language",\n'
        '    "Second key finding citing the technical metric name and explaining what it means in simple language",\n'
        '    "Third key finding (optional, up to 4 total)"\n'
        "  ],\n"
        '  "auditory_cues": "1-2 simple sentences describing what specific auditory phenomena a listener should pay attention to in playback"\n'
        "}"
    )

    explanations_text = "\n".join(f"- {exp}" for exp in explanations) if explanations else "None (Clean baseline)"

    user_prompt = (
        f"Acoustic Forensic Evaluation Data:\n"
        f"- Synthetic Evidence Index: {score:.1f}% / 100%\n"
        f"- Pipeline Decision: {decision_text}\n"
        f"- Evaluated Acoustic Domains: {domain_summary}\n\n"
        f"- Triggered Forensic Rules & Threshold Violations:\n{explanations_text}\n\n"
        "Synthesize these findings in simple language. For the observations that stood out the most, ensure you cite the technical metric names and explain them in simple terms as instructed. Provide your response strictly in the requested JSON format."
    )
    return system_prompt, user_prompt


def _extract_content_text(content: Any) -> str:
    """Extract plain text from string, dict, or list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(str(p.get("text", "")))
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)


def _parse_llm_response(raw_content: Any, provider: str) -> ForensicSummaryResult:
    """Parse and validate JSON response from the LLM with robust fallback."""
    text_content = _extract_content_text(raw_content)
    cleaned = _clean_json_text(text_content)
    try:
        data = json.loads(cleaned)
        headline = str(data.get("headline", "Forensic Acoustic Summary")).strip()
        overview = str(data.get("overview", "")).strip()
        raw_facts = data.get("standout_facts", [])
        if isinstance(raw_facts, list):
            standout_facts = tuple(str(f).strip() for f in raw_facts if str(f).strip())
        else:
            standout_facts = (str(raw_facts),)
        auditory_cues = str(data.get("auditory_cues", "")).strip()
    except Exception:
        # Fallback to extracting text directly from model response if JSON decoding failed
        headline = "AI Forensic Synthesis"
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        overview = lines[0] if lines else "AI analysis completed."
        bullet_facts = [
            line.lstrip("-*• ").strip()
            for line in lines
            if line.startswith(("-", "*", "•"))
        ]
        standout_facts = tuple(bullet_facts[:4]) if bullet_facts else ("Acoustic analysis processed by AI synthesis.",)
        auditory_cues = "Listen for unusual pitch stability or unnatural speech continuity."

    return ForensicSummaryResult(
        headline=headline,
        overview=overview,
        standout_facts=standout_facts,
        auditory_cues=auditory_cues,
        is_fallback=False,
        provider=provider,
    )


def _invoke_gemini_chain(
    score: float,
    decision_text: str,
    explanations: tuple[str, ...],
    domain_summary: str,
    api_key: str,
    model_name: str,
) -> ForensicSummaryResult:
    """Stream LangChain ChatGoogleGenerativeAI using Gemini API and assemble response."""
    # Suppress AFC warning from google.genai
    try:
        import google.genai.models
        google.genai.models.Models._logged_afc_warning = True
    except Exception:
        pass
    logging.getLogger("google.genai.models").setLevel(logging.ERROR)

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module=".*langchain_google_genai.*")
    warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    system_prompt, user_prompt = _get_prompts(score, decision_text, explanations, domain_summary)

    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
    }
    # Avoid passing temperature on models that use fixed sampling defaults (e.g. flash-lite / thinking)
    if not any(k in model_name.lower() for k in ("lite", "thinking", "reasoning")):
        kwargs["temperature"] = 0.2

    llm = ChatGoogleGenerativeAI(**kwargs)

    chunks: list[str] = []
    for chunk in llm.stream([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]):
        text_part = _extract_content_text(chunk.content)
        if text_part:
            chunks.append(text_part)

    full_text = "".join(chunks)
    return _parse_llm_response(full_text, provider="gemini")


def _invoke_openrouter_chain(
    score: float,
    decision_text: str,
    explanations: tuple[str, ...],
    domain_summary: str,
    api_key: str,
    model_name: str,
) -> ForensicSummaryResult:
    """Invoke LangChain ChatOpenAI via OpenRouter."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    system_prompt, user_prompt = _get_prompts(score, decision_text, explanations, domain_summary)

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        default_headers={
            "HTTP-Referer": "https://github.com/SignalGuard",
            "X-Title": "SignalGuard Audio Forensics",
        },
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    return _parse_llm_response(response.content, provider="openrouter")


def generate_forensic_summary(
    score: float,
    decision_text: str,
    explanations: tuple[str, ...] | list[str],
    domain_scores: tuple[Any, ...] | list[Any] | None = None,
) -> ForensicSummaryResult:
    """Generate an AI-powered human-understandable summary of acoustic findings.

    Prioritizes Google Gemini, falls back to OpenRouter, or provides structured
    heuristic interpretation if no API keys are present.
    """
    explanations_tuple = tuple(explanations)
    domain_summary = ""
    if domain_scores:
        summary_parts = []
        for d in domain_scores:
            name = getattr(d, "name", str(d))
            s = getattr(d, "score", 0.0)
            summary_parts.append(f"{name}: {s:.1f}")
        domain_summary = ", ".join(summary_parts)

    provider, api_key, model_name = resolve_llm_credentials()

    if not api_key or provider == "none":
        return _build_fallback_summary(
            score=score,
            decision_text=decision_text,
            explanations=explanations_tuple,
            error_message="Gemini API key not configured. Showing deterministic acoustic analysis.",
        )

    try:
        if provider == "gemini":
            return _invoke_gemini_chain(
                score=score,
                decision_text=decision_text,
                explanations=explanations_tuple,
                domain_summary=domain_summary,
                api_key=api_key,
                model_name=model_name,
            )
        else:
            return _invoke_openrouter_chain(
                score=score,
                decision_text=decision_text,
                explanations=explanations_tuple,
                domain_summary=domain_summary,
                api_key=api_key,
                model_name=model_name,
            )
    except Exception as exc:
        logger.warning("Failed to invoke %s LLM: %s", provider, exc)
        return _build_fallback_summary(
            score=score,
            decision_text=decision_text,
            explanations=explanations_tuple,
            error_message=f"AI synthesis unavailable ({exc.__class__.__name__}). Showing standard forensic findings.",
        )
