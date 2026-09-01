"""Deterministic short-time spectral evidence measurements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import librosa
import numpy as np
from numpy.typing import ArrayLike

from ._common import AnalysisResult, RuleEvaluation, coefficient_of_variation, detector_section, mono_audio, require_positive_integer


def analyze_spectral(samples: ArrayLike, sample_rate: int, *, config: Mapping[str, Any] | None = None) -> AnalysisResult:
    """Compute spectral shape and frame-to-frame stability metrics."""
    values, rate = mono_audio(samples, sample_rate)
    section = detector_section("spectral", config)
    n_fft = require_positive_integer(section["n_fft"], "spectral.n_fft")
    hop = require_positive_integer(section["hop_length"], "spectral.hop_length")
    if hop > n_fft:
        raise ValueError("spectral.hop_length must not exceed spectral.n_fft")
    if values.size < n_fft:
        return AnalysisResult("INSUFFICIENT_AUDIO", rate, {}, (), ("Audio is shorter than one spectral-analysis frame.",))
    if not np.any(values):
        return AnalysisResult(
            "SILENCE",
            rate,
            {"frame_count": 1 + (values.size - n_fft) // hop},
            (),
            ("Silent audio does not provide spectral forensic evidence.",),
        )
    magnitude = np.abs(librosa.stft(values, n_fft=n_fft, hop_length=hop, center=False, window="hann"))
    power = magnitude ** 2
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=rate)[0]
    flatness = librosa.feature.spectral_flatness(S=magnitude, power=2.0)[0]
    rolloff = librosa.feature.spectral_rolloff(S=power, sr=rate, roll_percent=float(section["rolloff_percent"]))[0]
    zcr = librosa.feature.zero_crossing_rate(values, frame_length=n_fft, hop_length=hop, center=False)[0]
    probabilities = power / np.maximum(np.sum(power, axis=0, keepdims=True), np.finfo(np.float64).tiny)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, np.finfo(np.float64).tiny)), axis=0) / np.log(power.shape[0])
    normalized_magnitude = magnitude / np.maximum(
        np.linalg.norm(magnitude, axis=0, keepdims=True),
        np.finfo(np.float64).tiny,
    )
    flux = (
        np.sqrt(np.sum(np.diff(normalized_magnitude, axis=1) ** 2, axis=0))
        if magnitude.shape[1] > 1
        else np.array([0.0])
    )
    metrics = {"mean_spectral_flatness": float(np.mean(flatness)), "mean_normalized_entropy": float(np.mean(entropy)), "mean_zero_crossing_rate": float(np.mean(zcr)), "centroid_cv": coefficient_of_variation(centroid), "bandwidth_cv": coefficient_of_variation(bandwidth), "flux_cv": coefficient_of_variation(flux), "rolloff_cv": coefficient_of_variation(rolloff), "frame_count": int(magnitude.shape[1])}
    rule_data = (("high_flatness", "mean_spectral_flatness", "high_flatness", ">"), ("high_entropy", "mean_normalized_entropy", "high_normalized_entropy", ">"), ("high_zcr", "mean_zero_crossing_rate", "high_zcr", ">"), ("low_centroid_cv", "centroid_cv", "low_centroid_cv", "<"), ("low_bandwidth_cv", "bandwidth_cv", "low_bandwidth_cv", "<"), ("low_flux_cv", "flux_cv", "low_flux_cv", "<"), ("low_rolloff_cv", "rolloff_cv", "low_rolloff_cv", "<"))
    rules = tuple(RuleEvaluation(code, metrics[key], float(section[limit]), comparison, None if metrics[key] is None else (metrics[key] > float(section[limit]) if comparison == ">" else metrics[key] < float(section[limit])), f"{key} crossed its configured spectral threshold.") for code, key, limit, comparison in rule_data)
    return AnalysisResult("OK", rate, metrics, rules)
