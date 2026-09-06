"""Boundary decay and abrupt cut-off analysis for voice forensics."""

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


def analyze_decay(
    samples: ArrayLike,
    sample_rate: int,
    *,
    config: Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Detect abrupt cut-offs and missing exponential room reverberation at phrase offsets.

    In physical spaces, vocal energy decays exponentially due to vocal-tract dissipation
    and room reverberation (typically 15-60 ms for 90%-to-10% energy collapse).
    AI synthesizers frequently truncate speech with unnatural step-function boundaries
    or rectangular synthesis windows (<8 ms collapse to dead silence).
    """
    values, rate = mono_audio(samples, sample_rate)
    section = detector_section("decay", config)
    abrupt_threshold_ms = require_finite_real(
        section.get("abrupt_decay_threshold_ms", 8.0),
        "decay.abrupt_decay_threshold_ms",
        minimum=1.0,
    )
    min_decay_ms_target = require_finite_real(
        section.get("min_natural_decay_ms", 12.0),
        "decay.min_natural_decay_ms",
        minimum=2.0,
    )

    duration_s = values.size / rate
    if duration_s < 0.25:
        return AnalysisResult(
            "INSUFFICIENT_AUDIO",
            rate,
            {"duration_seconds": duration_s},
            (),
            ("Audio is too short for boundary decay analysis.",),
        )

    energy = float(np.mean(values.astype(np.float64) ** 2))
    if energy <= np.finfo(np.float64).tiny or not np.any(values):
        return AnalysisResult(
            "SILENCE",
            rate,
            {"duration_seconds": duration_s},
            (),
            ("Silent audio contains no boundary offset events.",),
        )

    # Compute envelope via Hilbert analytic signal and 200 Hz smoothing
    analytic = scipy.signal.hilbert(values.astype(np.float64))
    envelope = np.abs(analytic)

    sos = scipy.signal.butter(2, min(200.0, 0.45 * rate), btype="lowpass", fs=rate, output="sos")
    smooth_env = scipy.signal.sosfiltfilt(sos, envelope)

    # Subsample to 1 ms resolution (1000 Hz) for decay time calculation
    subsample_factor = max(1, rate // 1000)
    res_rate = rate / subsample_factor
    env_1ms = smooth_env[::subsample_factor]

    p90_env = float(np.percentile(env_1ms, 90))
    p10_env = float(np.percentile(env_1ms, 10))

    if p90_env <= 1e-5 or p90_env - p10_env < 1e-4:
        return AnalysisResult(
            "LOW_DYNAMIC_RANGE",
            rate,
            {"duration_seconds": duration_s},
            (),
            ("Dynamic range too low to extract clear offset boundaries.",),
        )

    speech_level = p10_env + 0.50 * (p90_env - p10_env)

    # Find offsets: transitions from above speech_level to below
    is_above = env_1ms >= speech_level
    offset_indices: list[int] = []

    for i in range(1, len(is_above) - 1):
        if is_above[i - 1] and not is_above[i]:
            offset_indices.append(i)

    # If the file terminates while speech was active, add the end boundary
    if is_above[-1]:
        offset_indices.append(len(is_above) - 1)

    decay_times_ms: list[float] = []

    for idx in offset_indices:
        # Search backwards to find 90% peak of this speech burst
        search_back = max(0, idx - round(0.100 * res_rate))  # up to 100 ms back
        local_peak = float(np.max(env_1ms[search_back : idx + 1]))

        if local_peak < speech_level:
            continue

        target_90 = 0.90 * local_peak
        target_10 = 0.10 * local_peak

        # Find where it crosses 90%
        pos_90 = idx
        for k in range(idx, search_back, -1):
            if env_1ms[k] >= target_90:
                pos_90 = k
                break

        # Search forward for where it crosses 10%
        search_fwd = min(len(env_1ms), idx + round(0.150 * res_rate))
        pos_10 = search_fwd
        for k in range(idx, search_fwd):
            if env_1ms[k] <= target_10:
                pos_10 = k
                break

        decay_ms = float((pos_10 - pos_90) / res_rate * 1000.0)
        decay_times_ms.append(decay_ms)

    if not decay_times_ms:
        # Fallback to general boundary decay at end of utterance
        min_decay_ms = float(duration_s * 1000.0)
        abrupt_cutoffs = 0
    else:
        min_decay_ms = float(np.min(decay_times_ms))
        abrupt_cutoffs = int(np.sum(np.array(decay_times_ms) < abrupt_threshold_ms))

    # Rule: Abrupt cut-off (step function termination without reverberation)
    abrupt_triggered = bool(abrupt_cutoffs > 0 or min_decay_ms < abrupt_threshold_ms)
    abrupt_explanation = (
        f"Utterance offset terminates in {min_decay_ms:.1f} ms with an unnatural step-function cut-off."
        if abrupt_triggered
        else f"Boundary offset decay is natural ({min_decay_ms:.1f} ms)."
    )

    # Rule: Missing natural room decay tail
    unnatural_decay_triggered = bool(min_decay_ms < min_decay_ms_target)
    unnatural_decay_explanation = (
        f"Minimum boundary decay ({min_decay_ms:.1f} ms) is below natural room reverberation threshold."
        if unnatural_decay_triggered
        else "Reverberation tail decay profile is consistent with natural acoustics."
    )

    metrics: dict[str, float | int | None] = {
        "min_offset_decay_ms": min_decay_ms,
        "abrupt_cutoff_count": abrupt_cutoffs,
        "evaluated_offset_count": len(decay_times_ms),
    }

    rules = (
        RuleEvaluation(
            "abrupt_utterance_cutoff",
            min_decay_ms,
            abrupt_threshold_ms,
            "<",
            abrupt_triggered,
            abrupt_explanation,
        ),
        RuleEvaluation(
            "unnatural_decay_profile",
            min_decay_ms,
            min_decay_ms_target,
            "<",
            unnatural_decay_triggered,
            unnatural_decay_explanation,
        ),
    )

    return AnalysisResult("OK", rate, metrics, rules)
