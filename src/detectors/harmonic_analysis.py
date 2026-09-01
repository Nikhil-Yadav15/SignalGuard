"""F0-guided, deterministic harmonic-structure measurements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._common import AnalysisResult, RuleEvaluation, db_ratio, detector_section, frames, mono_audio, require_finite_real, require_positive_integer
from .f0_analysis import F0AnalysisResult, F0Settings, analyze_f0


def analyze_harmonics(samples: ArrayLike, sample_rate: int, *, f0_result: F0AnalysisResult | None = None, config: Mapping[str, Any] | None = None) -> AnalysisResult:
    """Measure energy concentrated near integer multiples of voiced F0.

    This is an energy accounting method, not a source-separation claim.  It
    deliberately skips unvoiced frames instead of inventing a fundamental.
    """
    values, rate = mono_audio(samples, sample_rate)
    section = detector_section("harmonic", config)
    frame_length = require_positive_integer(section["frame_length"], "harmonic.frame_length")
    hop = require_positive_integer(section["hop_length"], "harmonic.hop_length")
    count = require_positive_integer(section["harmonic_count"], "harmonic.harmonic_count")
    tolerance = require_finite_real(section["bin_tolerance_hz"], "harmonic.bin_tolerance_hz", minimum=0.0)
    if frame_length > values.size:
        return AnalysisResult("INSUFFICIENT_AUDIO", rate, {"voiced_frame_count": 0, "harmonic_energy_ratio": None, "out_of_harmonic_ratio": None, "harmonic_to_noise_ratio_db": None}, (), ("Audio is shorter than one harmonic-analysis frame.",))
    pitch = f0_result or analyze_f0(
        values,
        rate,
        settings=F0Settings.from_config(config),
    )
    if pitch.sample_rate_hz != rate:
        raise ValueError("f0_result sample rate does not match harmonic-analysis audio")
    pitch_by_start_sample = {
        int(round(time_seconds * rate)): f0
        for time_seconds, f0 in zip(
            pitch.track.frame_start_times_seconds,
            pitch.track.voiced_f0_hz,
        )
    }
    blocks = frames(values, frame_length, hop)
    window = np.hanning(frame_length)
    freq = np.fft.rfftfreq(frame_length, 1.0 / rate)
    ratios: list[float] = []
    hnrs: list[float] = []
    for index, block in enumerate(blocks):
        pitch_value = pitch_by_start_sample.get(index * hop)
        if pitch_value is None:
            continue
        f0 = float(pitch_value)
        power = np.abs(np.fft.rfft(block * window)) ** 2
        total = float(np.sum(power[1:]))
        if total <= np.finfo(np.float64).tiny:
            continue
        harmonic_bins = np.zeros(freq.size, dtype=bool)
        for harmonic in range(1, count + 1):
            target = harmonic * f0
            if target >= rate / 2:
                break
            harmonic_bins |= np.abs(freq - target) <= tolerance
        harmonic_energy = float(np.sum(power[harmonic_bins]))
        ratios.append(harmonic_energy / total)
        hnr_value = db_ratio(
            harmonic_energy,
            max(total - harmonic_energy, 0.0),
        )
        if hnr_value is not None:
            hnrs.append(hnr_value)
    harmonic_ratio = float(np.mean(ratios)) if ratios else None
    out_ratio = None if harmonic_ratio is None else float(1.0 - harmonic_ratio)
    hnr = float(np.mean(hnrs)) if hnrs else None
    rules = (
        RuleEvaluation("low_harmonic_energy_ratio", harmonic_ratio, float(section["low_harmonic_energy_ratio"]), "<", None if harmonic_ratio is None else harmonic_ratio < float(section["low_harmonic_energy_ratio"]), "Harmonic energy is below the configured concentration threshold."),
        RuleEvaluation("high_out_of_harmonic_ratio", out_ratio, float(section["high_out_of_harmonic_ratio"]), ">", None if out_ratio is None else out_ratio > float(section["high_out_of_harmonic_ratio"]), "Energy outside F0 harmonics is above the configured threshold."),
        RuleEvaluation("abnormal_hnr", hnr, float(section["hnr_low_db"]), "outside", None if hnr is None else hnr < float(section["hnr_low_db"]) or hnr > float(section["hnr_high_db"]), "Harmonic-to-residual ratio is outside the configured band."),
    )
    status = "OK" if ratios else "NO_VOICED_FRAMES"
    return AnalysisResult(status, rate, {"voiced_frame_count": len(ratios), "harmonic_energy_ratio": harmonic_ratio, "out_of_harmonic_ratio": out_ratio, "harmonic_to_noise_ratio_db": hnr}, rules, () if ratios else ("No voiced F0-aligned frames were available.",))
