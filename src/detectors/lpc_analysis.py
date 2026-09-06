"""LPC (Linear Predictive Coding) glottal residual analysis for voice forensics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import librosa
import numpy as np
import scipy.linalg
import scipy.signal
import scipy.stats
from numpy.typing import ArrayLike

from ._common import (
    AnalysisResult,
    RuleEvaluation,
    detector_section,
    mono_audio,
    require_finite_real,
    require_positive_integer,
)


def analyze_lpc(
    samples: ArrayLike,
    sample_rate: int,
    *,
    config: Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Extract LPC glottal excitation residual and evaluate physical pulse dynamics.

    Natural voiced human speech exhibits sharp, impulsive glottal closures with
    high excess kurtosis and asymmetric pulse shapes. Neural vocoders and synthetic
    generators often produce smeared, overly Gaussian glottal residuals with
    abnormally low kurtosis.
    """
    values, rate = mono_audio(samples, sample_rate)
    section = detector_section("lpc", config)
    order = require_positive_integer(section.get("order", 16), "lpc.order")
    low_kurtosis_threshold = require_finite_real(
        section.get("low_residual_kurtosis", 3.0),
        "lpc.low_residual_kurtosis",
    )
    low_gain_threshold = require_finite_real(
        section.get("low_prediction_gain_db", 3.0),
        "lpc.low_prediction_gain_db",
    )

    min_samples = order * 4
    if values.size < min_samples:
        return AnalysisResult(
            "INSUFFICIENT_AUDIO",
            rate,
            {},
            (),
            ("Audio is too short for LPC analysis.",),
        )

    energy = float(np.mean(values.astype(np.float64) ** 2))
    if energy <= np.finfo(np.float64).tiny or not np.any(values):
        return AnalysisResult(
            "SILENCE",
            rate,
            {},
            (),
            ("Silent audio contains no glottal excitation evidence.",),
        )

    # Compute LPC coefficients on pre-emphasized signal for stable pole estimation
    pre_emphasis = 0.97
    pre_emphasized = np.append(values[0], values[1:] - pre_emphasis * values[:-1])

    # Extract LPC coefficients
    try:
        a_coeffs = librosa.lpc(pre_emphasized.astype(np.float64), order=order)
        # Inverse-filter to obtain residual excitation e[n] = s[n] - sum(a_k * s[n-k])
        # In scipy.signal.lfilter with b=a_coeffs and a=1.0:
        residual = scipy.signal.lfilter(a_coeffs, [1.0], values.astype(np.float64))
    except (FloatingPointError, ValueError, scipy.linalg.LinAlgError) as exc:
        return AnalysisResult(
            "ERROR",
            rate,
            {},
            (),
            (f"LPC estimation failed: {exc}",),
        )

    # Residual dynamics
    res_var = float(np.var(residual))
    sig_var = float(np.var(values))
    tiny = np.finfo(np.float64).tiny

    if res_var <= tiny:
        prediction_gain_db = 0.0
    else:
        prediction_gain_db = float(10.0 * np.log10(max(sig_var, tiny) / max(res_var, tiny)))

    # Kurtosis: Fisher's definition (normal distribution = 0.0)
    # Impulsive human glottal closures have high positive excess kurtosis (> 3.0)
    # Smeared synthetic vocoder residuals have lower kurtosis (< 3.0 or near Gaussian)
    residual_kurtosis = float(scipy.stats.kurtosis(residual, fisher=True))
    residual_skewness = float(scipy.stats.skew(residual))

    metrics: dict[str, float | int | None] = {
        "residual_kurtosis": residual_kurtosis,
        "prediction_gain_db": prediction_gain_db,
        "residual_skewness": residual_skewness,
        "lpc_order": order,
    }

    rules = (
        RuleEvaluation(
            "low_residual_kurtosis",
            residual_kurtosis,
            low_kurtosis_threshold,
            "<",
            residual_kurtosis < low_kurtosis_threshold,
            "Glottal residual excess kurtosis is abnormally low (smeared vocoder pulses).",
        ),
        RuleEvaluation(
            "low_prediction_gain",
            prediction_gain_db,
            low_gain_threshold,
            "<",
            prediction_gain_db < low_gain_threshold,
            "LPC prediction gain is below expected vocal-tract resonance levels.",
        ),
    )

    return AnalysisResult("OK", rate, metrics, rules)
