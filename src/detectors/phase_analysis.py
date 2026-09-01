"""Phase-continuity measurements from a deterministic STFT."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import librosa
import numpy as np
from numpy.typing import ArrayLike

from ._common import (
    AnalysisResult,
    RuleEvaluation,
    detector_section,
    mono_audio,
    require_finite_real,
    require_positive_integer,
)


def analyze_phase(
    samples: ArrayLike,
    sample_rate: int,
    *,
    config: Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Measure residual phase motion after removing normal STFT phase advance."""

    values, rate = mono_audio(samples, sample_rate)
    section = detector_section("phase", config)
    n_fft = require_positive_integer(section["n_fft"], "phase.n_fft")
    hop = require_positive_integer(section["hop_length"], "phase.hop_length")
    if hop > n_fft:
        raise ValueError("phase.hop_length must not exceed phase.n_fft")

    jump_threshold = require_finite_real(
        section["phase_jump_radians"],
        "phase.phase_jump_radians",
        minimum=0.0,
    )
    magnitude_ratio = require_finite_real(
        section["minimum_bin_magnitude_ratio"],
        "phase.minimum_bin_magnitude_ratio",
        minimum=0.0,
    )
    if magnitude_ratio > 1.0:
        raise ValueError("phase.minimum_bin_magnitude_ratio must not exceed 1")

    if values.size < n_fft:
        return AnalysisResult(
            "INSUFFICIENT_AUDIO",
            rate,
            {},
            (),
            ("Audio is shorter than one phase-analysis frame.",),
        )
    if not np.any(values):
        return AnalysisResult(
            "SILENCE",
            rate,
            {"frame_count": 1 + (values.size - n_fft) // hop},
            (),
            ("Silent audio does not provide phase forensic evidence.",),
        )

    spectrum = librosa.stft(
        values,
        n_fft=n_fft,
        hop_length=hop,
        center=False,
        window="hann",
    )
    if spectrum.shape[1] < 2:
        return AnalysisResult(
            "INSUFFICIENT_FRAMES",
            rate,
            {},
            (),
            ("At least two STFT frames are required for phase evidence.",),
        )

    phase = np.angle(spectrum)
    magnitude = np.abs(spectrum)
    frame_peak = np.max(magnitude, axis=0, keepdims=True)
    significant = magnitude >= magnitude_ratio * np.maximum(
        frame_peak,
        np.finfo(np.float64).tiny,
    )

    # A sinusoid advances by 2*pi*k*hop/n_fft at bin k even when perfectly
    # stable.  Remove that deterministic advance and wrap the residual.
    bin_indexes = np.arange(spectrum.shape[0], dtype=np.float64)[:, None]
    expected_advance = 2.0 * np.pi * hop * bin_indexes / n_fft
    residual = _wrap_phase(np.diff(phase, axis=1) - expected_advance)
    residual_mask = significant[:, 1:] & significant[:, :-1]
    residual_values = np.abs(residual[residual_mask])

    if residual_values.size:
        phase_jump_ratio = float(np.mean(residual_values >= jump_threshold))
        variation = float(np.mean(residual_values))
    else:
        phase_jump_ratio = 0.0
        variation = 0.0

    # Group delay is the negative phase derivative across frequency.  Use a
    # robust median/MAD z-score and only count adjacent significant bins.
    group_delay = _wrap_phase(-np.diff(phase, axis=0))
    group_mask = significant[1:, :] & significant[:-1, :]
    group_values = group_delay[group_mask]
    outlier_ratio = _robust_outlier_ratio(
        group_values,
        require_finite_real(
            section["group_delay_outlier_zscore"],
            "phase.group_delay_outlier_zscore",
            minimum=0.0,
        ),
    )

    high_jump_ratio = float(section["high_phase_jump_ratio"])
    low_variation = float(section["low_phase_variation"])
    high_group_delay_ratio = float(section["high_group_delay_outlier_ratio"])
    metrics: dict[str, float | int | None] = {
        "phase_jump_ratio": phase_jump_ratio,
        "mean_phase_variation_radians": variation,
        "group_delay_outlier_ratio": outlier_ratio,
        "significant_phase_sample_count": int(residual_values.size),
        "frame_count": int(spectrum.shape[1]),
    }
    rules = (
        RuleEvaluation(
            "high_phase_jump_ratio",
            phase_jump_ratio,
            high_jump_ratio,
            ">",
            phase_jump_ratio > high_jump_ratio,
            "Residual phase jumps exceed the configured ratio.",
        ),
        RuleEvaluation(
            "low_phase_variation",
            variation,
            low_variation,
            "<",
            variation < low_variation,
            "Residual frame-to-frame phase variation is unusually low.",
        ),
        RuleEvaluation(
            "group_delay_outliers",
            outlier_ratio,
            high_group_delay_ratio,
            ">",
            outlier_ratio > high_group_delay_ratio,
            "Group-delay outliers exceed the configured ratio.",
        ),
    )
    return AnalysisResult("OK", rate, metrics, rules)


def _wrap_phase(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _robust_outlier_ratio(values: np.ndarray, z_limit: float) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    absolute_deviation = np.abs(values - median)
    mad = float(np.median(absolute_deviation))
    if mad <= np.finfo(np.float64).tiny:
        return 0.0
    robust_z = 0.6744897501960817 * absolute_deviation / mad
    return float(np.mean(robust_z >= z_limit))
