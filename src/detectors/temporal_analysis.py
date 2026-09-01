"""Cross-segment temporal consistency analysis."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from ._common import AnalysisResult, RuleEvaluation, coefficient_of_variation, detector_section, mono_audio, require_positive_integer
from .f0_analysis import F0AnalysisResult


def analyze_temporal(segments: Sequence[ArrayLike], sample_rate: int, *, f0_results: Sequence[F0AnalysisResult] | None = None, config: Mapping[str, Any] | None = None) -> AnalysisResult:
    """Compare RMS, centroid, ZCR, and F0 summaries across clip segments."""
    _, rate = mono_audio(np.asarray([0.0], dtype=np.float32), sample_rate)
    section = detector_section("temporal", config)
    minimum_segments = require_positive_integer(
        section["min_segments"],
        "temporal.min_segments",
    )
    if f0_results is not None and len(f0_results) != len(segments):
        raise ValueError("f0_results must align one-to-one with segments")
    if len(segments) < minimum_segments:
        return AnalysisResult("INSUFFICIENT_SEGMENTS", rate, {"segment_count": len(segments)}, (), ("Not enough segments for temporal consistency analysis.",))
    energy: list[float] = []
    centroid: list[float] = []
    zcr: list[float] = []
    for segment in segments:
        values, _ = mono_audio(segment, rate)
        energy.append(float(np.sqrt(np.mean(values.astype(np.float64) ** 2))))
        if values.size >= 2:
            magnitude = np.abs(np.fft.rfft(values * np.hanning(values.size)))
            frequencies = np.fft.rfftfreq(values.size, 1 / rate)
            centroid.append(float(np.sum(frequencies * magnitude) / max(np.sum(magnitude), np.finfo(float).tiny)))
            zcr.append(float(np.mean(values[:-1] * values[1:] < 0)))
        else:
            centroid.append(0.0); zcr.append(0.0)
    if not any(value > np.finfo(np.float64).tiny for value in energy):
        return AnalysisResult(
            "SILENCE",
            rate,
            {"segment_count": len(segments)},
            (),
            ("Silent segments do not provide temporal forensic evidence.",),
        )
    f0_values = [] if f0_results is None else [result.metrics.mean_f0_hz for result in f0_results if result.metrics.mean_f0_hz is not None]
    metrics = {"segment_count": len(segments), "energy_cv": coefficient_of_variation(energy), "centroid_cv": coefficient_of_variation(centroid), "zcr_cv": coefficient_of_variation(zcr), "f0_cv": coefficient_of_variation(f0_values)}
    pairs = (("low_energy_cv", "energy_cv", "low_energy_cv"), ("low_centroid_cv", "centroid_cv", "low_centroid_cv"), ("low_zcr_cv", "zcr_cv", "low_zcr_cv"), ("low_f0_cv", "f0_cv", "low_f0_cv"))
    rules = tuple(RuleEvaluation(code, metrics[key], float(section[limit]), "<", None if metrics[key] is None else metrics[key] < float(section[limit]), f"{key} is below its temporal-variation threshold.") for code, key, limit in pairs)
    return AnalysisResult("OK", rate, metrics, rules)
