"""SignalGuard LLM explainer package for human-understandable acoustic forensics."""

from .explainer import (
    ForensicSummaryResult,
    generate_forensic_summary,
    resolve_llm_credentials,
    resolve_openrouter_credentials,
)

__all__ = [
    "ForensicSummaryResult",
    "generate_forensic_summary",
    "resolve_llm_credentials",
    "resolve_openrouter_credentials",
]
