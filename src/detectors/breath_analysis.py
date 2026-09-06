"""Breath, pause, and respiration dynamics analysis for synthetic speech detection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._common import (
    AnalysisResult,
    RuleEvaluation,
    detector_section,
    mono_audio,
    require_finite_real,
)


def analyze_breath(
    samples: ArrayLike,
    sample_rate: int,
    *,
    config: Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Analyze respiration pauses and digital silence gaps in continuous speech.

    Natural human speech contains involuntary respiratory micro-pauses (inhaling air
    every 2-4 seconds) and continuous room acoustic presence. AI-generated voices
    frequently suffer from two respiratory extremes:
    1. Run-on speech: Continuous phonation lasting >4.5 seconds with zero pauses.
    2. Digital silence: Speech gaps that drop into unnatural mathematical dead space
       (<-75 dBFS or absolute zeros) lacking ambient room acoustics.
    """
    values, rate = mono_audio(samples, sample_rate)
    section = detector_section("breath", config)
    max_burst_s = require_finite_real(
        section.get("max_continuous_speech_seconds", 4.5),
        "breath.max_continuous_speech_seconds",
        minimum=1.0,
    )
    silence_floor_dbfs = require_finite_real(
        section.get("digital_silence_floor_dbfs", -75.0),
        "breath.digital_silence_floor_dbfs",
        minimum=-120.0,
    )
    max_silence_ratio = require_finite_real(
        section.get("high_digital_silence_ratio", 0.20),
        "breath.high_digital_silence_ratio",
        minimum=0.0,
    )

    duration_s = values.size / rate
    if duration_s < 0.5:
        return AnalysisResult(
            "INSUFFICIENT_AUDIO",
            rate,
            {"duration_seconds": duration_s},
            (),
            ("Audio is too short for breath and pause analysis.",),
        )

    energy = float(np.mean(values.astype(np.float64) ** 2))
    if energy <= np.finfo(np.float64).tiny or not np.any(values):
        return AnalysisResult(
            "SILENCE",
            rate,
            {"duration_seconds": duration_s},
            (),
            ("Silent audio contains no speech respiration evidence.",),
        )

    # Frame energy in 20 ms frames with 10 ms hop
    frame_len = max(16, round(rate * 0.020))
    hop_len = max(8, round(rate * 0.010))
    num_frames = 1 + (values.size - frame_len) // hop_len

    if num_frames < 4:
        return AnalysisResult("INSUFFICIENT_FRAMES", rate, {}, (), ("Too few frames.",))

    indices = np.arange(num_frames)[:, None] * hop_len + np.arange(frame_len)[None, :]
    framed = values[indices].astype(np.float64)
    frame_powers = np.mean(framed ** 2, axis=1)

    tiny = np.finfo(np.float64).tiny
    frame_db = 10.0 * np.log10(np.maximum(frame_powers, tiny))

    # Adaptive speech threshold relative to active speech peak
    p90_db = float(np.percentile(frame_db, 90))
    speech_threshold_db = max(-55.0, p90_db - 30.0)

    is_speech = frame_db >= speech_threshold_db

    # Find continuous speech bursts
    speech_runs: list[float] = []
    current_run = 0
    for flag in is_speech:
        if flag:
            current_run += 1
        else:
            if current_run > 0:
                speech_runs.append(current_run * (hop_len / rate))
                current_run = 0
    if current_run > 0:
        speech_runs.append(current_run * (hop_len / rate))

    max_speech_burst = max(speech_runs) if speech_runs else 0.0

    # Find qualifying inter-phrase pauses (>= 150 ms sub-threshold)
    min_pause_frames = round(0.150 / (hop_len / rate))
    pause_frames: list[int] = []
    current_pause = 0
    pause_start = 0

    for i, flag in enumerate(is_speech):
        if not flag:
            if current_pause == 0:
                pause_start = i
            current_pause += 1
        else:
            if current_pause >= min_pause_frames:
                pause_frames.extend(range(pause_start, pause_start + current_pause))
            current_pause = 0
    if current_pause >= min_pause_frames:
        pause_frames.extend(range(pause_start, pause_start + current_pause))

    # Calculate digital silence ratio within pauses
    if pause_frames:
        pause_powers_db = frame_db[pause_frames]
        dead_frames = np.sum(pause_powers_db <= silence_floor_dbfs)
        digital_silence_ratio = float(dead_frames / len(pause_frames))
        pause_count = int(np.ceil(len(pause_frames) / max(1, min_pause_frames)))
    else:
        digital_silence_ratio = 0.0
        pause_count = 0

    # Rule 1: Run-on speech (only evaluated if audio is long enough to observe bursts > max_burst_s)
    if duration_s >= max_burst_s + 0.5:
        run_on_triggered = bool(max_speech_burst > max_burst_s)
        run_on_explanation = (
            f"Speech runs continuously for {max_speech_burst:.1f}s without natural respiratory pauses."
            if run_on_triggered
            else "Speech pauses conform to natural respiratory limits."
        )
    else:
        run_on_triggered = False
        run_on_explanation = "Audio duration is shorter than human run-on respiratory threshold."

    # Rule 2: Digital dead silence in pauses
    if pause_count > 0:
        silence_triggered = bool(digital_silence_ratio > max_silence_ratio)
        silence_explanation = (
            f"Speech gaps contain {digital_silence_ratio * 100:.1f}% absolute digital silence lacking room presence."
            if silence_triggered
            else "Pause acoustic presence is natural."
        )
    else:
        silence_triggered = False
        silence_explanation = "No inter-phrase pause gaps observed."

    metrics: dict[str, float | int | None] = {
        "max_speech_burst_seconds": float(max_speech_burst),
        "digital_silence_ratio": float(digital_silence_ratio),
        "pause_count": int(pause_count),
        "duration_seconds": float(duration_s),
    }

    rules = (
        RuleEvaluation(
            "run_on_speech",
            max_speech_burst,
            max_burst_s,
            ">",
            run_on_triggered,
            run_on_explanation,
        ),
        RuleEvaluation(
            "digital_silence_pauses",
            digital_silence_ratio,
            max_silence_ratio,
            ">",
            silence_triggered,
            silence_explanation,
        ),
    )

    return AnalysisResult("OK", rate, metrics, rules)
