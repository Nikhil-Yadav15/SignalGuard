"""Bispectrum and Higher-Order Spectral Analysis (HOSA) for voice forensics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._common import (
    AnalysisResult,
    RuleEvaluation,
    detector_section,
    frames,
    mono_audio,
    require_finite_real,
    require_positive_integer,
)


def analyze_bispectrum(
    samples: ArrayLike,
    sample_rate: int,
    *,
    config: Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Measure quadratic phase coupling (QPC) between vocal tract harmonics.

    Natural vocal tract aerodynamics produce non-linear interactions and strong
    quadratic phase coupling (QPC) between fundamental frequency harmonics.
    Synthesizers and neural vocoders often synthesize harmonic bands independently
    or with random phase, resulting in abnormally low bicoherence.
    """
    values, rate = mono_audio(samples, sample_rate)
    section = detector_section("bispectrum", config)
    n_fft = require_positive_integer(section.get("n_fft", 512), "bispectrum.n_fft")
    hop_length = require_positive_integer(section.get("hop_length", 256), "bispectrum.hop_length")
    fmax_hz = require_finite_real(section.get("fmax_hz", 2000.0), "bispectrum.fmax_hz", minimum=100.0)
    low_bicoherence_threshold = require_finite_real(
        section.get("low_bicoherence", 0.08),
        "bispectrum.low_bicoherence",
    )
    low_max_threshold = require_finite_real(
        section.get("low_max_bicoherence", 0.35),
        "bispectrum.low_max_bicoherence",
    )

    if values.size < n_fft:
        return AnalysisResult(
            "INSUFFICIENT_AUDIO",
            rate,
            {},
            (),
            ("Audio is shorter than one bispectrum analysis frame.",),
        )

    energy = float(np.mean(values.astype(np.float64) ** 2))
    if energy <= np.finfo(np.float64).tiny or not np.any(values):
        return AnalysisResult(
            "SILENCE",
            rate,
            {},
            (),
            ("Silent audio does not provide phase coupling evidence.",),
        )

    raw_frames = frames(values, n_fft, hop_length)
    if raw_frames.shape[0] < 4:
        return AnalysisResult(
            "INSUFFICIENT_FRAMES",
            rate,
            {"frame_count": int(raw_frames.shape[0])},
            (),
            ("At least 4 frames are required for statistical bispectrum averaging.",),
        )

    # Window frames and compute STFT: shape (K_frames, N_bins)
    window = np.hanning(n_fft).astype(np.float64)
    stft = np.fft.rfft(raw_frames * window[None, :], axis=1)
    k_frames, total_bins = stft.shape

    # Focus on the vocal harmonic band up to fmax_hz
    freq_resolution = rate / n_fft
    max_bin = min(total_bins - 1, int(np.floor(fmax_hz / freq_resolution)))
    min_bin = max(1, int(np.floor(80.0 / freq_resolution)))  # Skip DC and sub-speech

    if max_bin <= min_bin + 2:
        return AnalysisResult("ERROR", rate, {}, (), ("Frequency range too narrow.",))

    # Select candidate bifrequencies (f1, f2) such that f1 + f2 <= max_bin
    f1_indices, f2_indices = np.meshgrid(
        np.arange(min_bin, max_bin),
        np.arange(min_bin, max_bin),
        indexing="ij",
    )
    valid_mask = (f1_indices >= f2_indices) & ((f1_indices + f2_indices) <= max_bin)
    f1_flat = f1_indices[valid_mask]
    f2_flat = f2_indices[valid_mask]
    sum_flat = f1_flat + f2_flat

    # X(f1), X(f2), X*(f1+f2): each shape (K_frames, N_pairs)
    x1 = stft[:, f1_flat]
    x2 = stft[:, f2_flat]
    x3_conj = np.conj(stft[:, sum_flat])

    # Bispectrum B = E[X1 * X2 * X3*]
    bispectrum = np.mean(x1 * x2 * x3_conj, axis=0)

    # Normalization terms for squared bicoherence
    p = np.mean(np.abs(x1 * x2) ** 2, axis=0)
    q = np.mean(np.abs(x3_conj) ** 2, axis=0)
    denom = np.sqrt(p * q)

    tiny = np.finfo(np.float64).tiny
    bicoherence = np.abs(bispectrum) / np.maximum(denom, tiny)
    # Clip bicoherence to [0, 1] due to finite sample estimation variance
    bicoherence = np.clip(bicoherence, 0.0, 1.0)

    mean_bic = float(np.mean(bicoherence)) if bicoherence.size > 0 else 0.0
    max_bic = float(np.max(bicoherence)) if bicoherence.size > 0 else 0.0

    metrics: dict[str, float | int | None] = {
        "mean_bicoherence": mean_bic,
        "max_bicoherence": max_bic,
        "evaluated_pair_count": int(bicoherence.size),
        "frame_count": int(k_frames),
    }

    rules = (
        RuleEvaluation(
            "low_bicoherence",
            mean_bic,
            low_bicoherence_threshold,
            "<",
            mean_bic < low_bicoherence_threshold,
            "Mean bicoherence is below human vocal quadratic phase coupling threshold.",
        ),
        RuleEvaluation(
            "low_max_bicoherence",
            max_bic,
            low_max_threshold,
            "<",
            max_bic < low_max_threshold,
            "Peak harmonic phase coupling is abnormally weak or absent.",
        ),
    )

    return AnalysisResult("OK", rate, metrics, rules)
