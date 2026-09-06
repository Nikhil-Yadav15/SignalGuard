"""Modulation spectral analysis for speech rhythm and prosodic envelope dynamics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.signal
from numpy.typing import ArrayLike

from ._common import (
    AnalysisResult,
    RuleEvaluation,
    detector_section,
    mono_audio,
    require_finite_real,
)


def analyze_modulation(
    samples: ArrayLike,
    sample_rate: int,
    *,
    config: Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Analyze temporal envelope modulations in the syllabic rate domain (2–8 Hz).

    Natural human speech conveys linguistic and prosodic information through slow
    amplitude variations corresponding to syllabic rates (2–8 Hz). Synthetic speech
    often exhibits either mechanical, hyper-regular modulation spikes (robotic rhythm)
    or unnaturally flattened envelope dynamics with deficient modulation depth.
    """
    values, rate = mono_audio(samples, sample_rate)
    section = detector_section("modulation", config)
    min_duration_s = require_finite_real(
        section.get("min_duration_seconds", 0.5),
        "modulation.min_duration_seconds",
        minimum=0.2,
    )
    max_peak_ratio = require_finite_real(
        section.get("high_modulation_peak_ratio", 6.0),
        "modulation.high_modulation_peak_ratio",
        minimum=1.0,
    )
    low_depth_threshold = require_finite_real(
        section.get("low_modulation_depth", 0.20),
        "modulation.low_modulation_depth",
        minimum=0.0,
    )

    duration_s = values.size / rate
    if duration_s < min_duration_s:
        return AnalysisResult(
            "INSUFFICIENT_AUDIO",
            rate,
            {"duration_seconds": duration_s},
            (),
            ("Audio is shorter than the minimum required for modulation analysis.",),
        )

    energy = float(np.mean(values.astype(np.float64) ** 2))
    if energy <= np.finfo(np.float64).tiny or not np.any(values):
        return AnalysisResult(
            "SILENCE",
            rate,
            {"duration_seconds": duration_s},
            (),
            ("Silent audio contains no modulation dynamics.",),
        )

    # Extract temporal envelope via Hilbert analytic signal
    analytic = scipy.signal.hilbert(values.astype(np.float64))
    envelope = np.abs(analytic)

    # Downsample envelope to ~100 Hz frame rate for efficient modulation FFT
    target_env_rate = 100
    downsample_factor = max(1, rate // target_env_rate)
    effective_env_rate = rate / downsample_factor

    # Anti-aliasing filter before decimation: low-pass at 30 Hz
    cutoff_hz = min(30.0, 0.45 * effective_env_rate)
    sos = scipy.signal.butter(4, cutoff_hz, btype="lowpass", fs=rate, output="sos")
    filtered_env = scipy.signal.sosfiltfilt(sos, envelope)
    decimated_env = filtered_env[::downsample_factor]

    if decimated_env.size < 8:
        return AnalysisResult(
            "INSUFFICIENT_FRAMES",
            rate,
            {"envelope_frames": int(decimated_env.size)},
            (),
            ("Decimated envelope has too few samples.",),
        )

    # Modulation depth: dynamic range using robust percentiles
    p95 = float(np.percentile(decimated_env, 95))
    p5 = float(np.percentile(decimated_env, 5))
    modulation_depth = (p95 - p5) / max(p95 + p5, 1e-6)

    # Modulation spectrum: FFT of zero-mean windowed envelope
    centered_env = decimated_env - np.mean(decimated_env)
    env_window = np.hanning(centered_env.size)
    mod_fft = np.abs(np.fft.rfft(centered_env * env_window))
    mod_freqs = np.fft.rfftfreq(centered_env.size, d=1.0 / effective_env_rate)

    # Syllabic band: 2.0 to 8.0 Hz
    # Overall speech modulation band: 0.5 to 20.0 Hz
    total_mask = (mod_freqs >= 0.5) & (mod_freqs <= 20.0)
    syllabic_mask = (mod_freqs >= 2.0) & (mod_freqs <= 8.0)

    total_energy = float(np.sum(mod_fft[total_mask] ** 2))
    syllabic_energy = float(np.sum(mod_fft[syllabic_mask] ** 2))

    tiny = np.finfo(np.float64).tiny
    syllabic_energy_ratio = syllabic_energy / max(total_energy, tiny)

    # Modulation peak ratio: ratio of max peak bin to average in the modulation band
    if np.any(total_mask):
        band_fft = mod_fft[total_mask]
        peak_energy = float(np.max(band_fft))
        mean_energy = float(np.mean(band_fft))
        modulation_peak_ratio = peak_energy / max(mean_energy, tiny)
    else:
        modulation_peak_ratio = 1.0

    metrics: dict[str, float | int | None] = {
        "syllabic_energy_ratio": float(syllabic_energy_ratio),
        "modulation_depth": float(modulation_depth),
        "modulation_peak_ratio": float(modulation_peak_ratio),
        "envelope_rate_hz": float(effective_env_rate),
    }

    rules = (
        RuleEvaluation(
            "unnatural_modulation_peak",
            modulation_peak_ratio,
            max_peak_ratio,
            ">",
            modulation_peak_ratio > max_peak_ratio,
            "Hyper-regular modulation peak indicates robotic or clock-like cadence.",
        ),
        RuleEvaluation(
            "low_modulation_depth",
            modulation_depth,
            low_depth_threshold,
            "<",
            modulation_depth < low_depth_threshold,
            "Temporal envelope modulation depth is abnormally flat or suppressed.",
        ),
    )

    return AnalysisResult("OK", rate, metrics, rules)
