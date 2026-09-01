"""Small shared primitives for deterministic forensic detectors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..preprocessing import pcm_to_float
from ..utils import ConfigError, get_config_section, load_config, to_json_safe, validate_audio_array, validate_sample_rate


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    """A configuration threshold comparison, suitable for score explanations."""

    code: str
    observed_value: float | None
    threshold: float
    comparison: str
    triggered: bool | None
    explanation: str


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Shared JSON-safe envelope used by non-F0 detector modules."""

    status: str
    sample_rate_hz: int
    metrics: dict[str, float | int | None]
    rule_evaluations: tuple[RuleEvaluation, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe(self)


def detector_section(name: str, config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    root = load_config() if config is None else config
    return get_config_section(get_config_section(root, "forensics"), name)


def mono_audio(samples: ArrayLike, sample_rate: int, *, allow_empty: bool = False) -> tuple[NDArray[np.float32], int]:
    rate = validate_sample_rate(sample_rate)
    raw = validate_audio_array(samples, allow_empty=allow_empty, allowed_dimensions=(1,))
    return pcm_to_float(raw), rate


def require_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def require_finite_real(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        relation = f" at least {minimum:g}" if minimum is not None else " finite"
        raise ValueError(f"{name} must be{relation}")
    return result


def frames(values: NDArray[np.float32], frame_length: int, hop_length: int) -> NDArray[np.float64]:
    """Return complete overlapping frames without implicit padding."""
    if values.size < frame_length:
        return np.empty((0, frame_length), dtype=np.float64)
    count = 1 + (values.size - frame_length) // hop_length
    starts = np.arange(count)[:, None] * hop_length
    indexes = starts + np.arange(frame_length)[None, :]
    return values[indexes].astype(np.float64, copy=False)


def coefficient_of_variation(values: ArrayLike) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    mean = float(np.mean(array))
    if abs(mean) <= np.finfo(np.float64).tiny:
        return 0.0 if np.allclose(array, 0.0) else None
    return float(np.std(array) / abs(mean))


def db_ratio(numerator: float, denominator: float) -> float | None:
    if numerator < 0 or denominator < 0 or not np.isfinite(numerator + denominator):
        return None
    floor = np.finfo(np.float64).tiny
    return float(10.0 * np.log10(max(numerator, floor) / max(denominator, floor)))
