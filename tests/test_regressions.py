"""Regression coverage for cross-module bugs found during the full audit."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
import soundfile as sf

from src.dataset_generator import generate_clean_speech
from src.detectors import (
    analyze_phase,
    analyze_spectral,
    analyze_temporal,
    detect_corruption,
)
from src.pipeline import PipelineDecision, SignalGuardPipeline
from src.quality import validate_restoration
from src.restoration import notch_hum, repair_dropouts
from src.utils import ConfigError, load_config
from src.visualization import plot_spectrogram


RATE = 16_000


def _tone(frequency: float, seconds: float = 1.0) -> np.ndarray:
    time = np.arange(round(RATE * seconds)) / RATE
    return (0.5 * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


def test_pipeline_propagates_and_isolates_custom_configuration() -> None:
    config = deepcopy(load_config())
    config["preprocessing"]["target_peak"] = 0.10
    config["forensics"]["f0"]["hop_length_samples"] = 512
    pipeline = SignalGuardPipeline(config)
    config["preprocessing"]["target_peak"] = 0.90
    result = pipeline.analyze_samples(_tone(200.0), RATE)
    assert np.max(np.abs(result.audio.samples)) == pytest.approx(0.10, abs=1e-6)
    assert result.forensic_results["f0"].hop_length_samples == 512


def test_pipeline_rejects_invalid_custom_configuration() -> None:
    config = deepcopy(load_config())
    config["preprocessing"]["target_peak"] = 2.0
    with pytest.raises(ConfigError, match="target_peak"):
        SignalGuardPipeline(config)


def test_complete_pipeline_result_is_strict_json() -> None:
    result = SignalGuardPipeline().analyze_samples(_tone(220.0, 0.5), RATE)
    encoded = json.dumps(result.to_dict(), allow_nan=False)
    assert '"decision"' in encoded
    assert len(result.to_dict()["audio"]["samples"]) == result.audio.samples.size


def test_silence_does_not_generate_spectral_phase_or_temporal_rules() -> None:
    silence = np.zeros(RATE, dtype=np.float32)
    spectral = analyze_spectral(silence, RATE)
    phase = analyze_phase(silence, RATE)
    temporal = analyze_temporal([silence, silence], RATE)
    assert (spectral.status, phase.status, temporal.status) == (
        "SILENCE",
        "SILENCE",
        "SILENCE",
    )
    assert not spectral.rule_evaluations
    assert not phase.rule_evaluations
    assert not temporal.rule_evaluations


def test_phase_analysis_removes_expected_stft_bin_advance() -> None:
    result = analyze_phase(_tone(250.0), RATE)
    jump_rule = next(
        rule
        for rule in result.rule_evaluations
        if rule.code == "high_phase_jump_ratio"
    )
    group_rule = next(
        rule
        for rule in result.rule_evaluations
        if rule.code == "group_delay_outliers"
    )
    assert jump_rule.triggered is False
    assert group_rule.threshold > 0


def test_clipping_requires_a_qualified_flat_top_run() -> None:
    isolated = _tone(190.0)
    isolated[1000] = 1.0
    isolated_report = detect_corruption(isolated, RATE)
    assert "clipping" not in isolated_report.detected_kinds
    assert isolated_report.metrics["flat_top_run_count"] == 0

    flat_top = _tone(190.0)
    flat_top[2000:2100] = 1.0
    flat_report = detect_corruption(flat_top, RATE)
    assert "clipping" in flat_report.detected_kinds
    assert flat_report.metrics["flat_top_run_count"] == 1


def test_smooth_full_scale_sine_is_not_clipping_or_impulse_corruption() -> None:
    time = np.arange(1023) / RATE
    signal = np.sin(2 * np.pi * 200 * time).astype(np.float32)
    report = detect_corruption(signal, RATE)
    assert "clipping" not in report.detected_kinds
    assert "impulse" not in report.detected_kinds


def test_short_audio_has_no_empty_fft_band_warnings() -> None:
    signal = _tone(200.0)[:100]
    report = detect_corruption(signal, RATE)
    assert report.metrics["hum_prominence_db"] is None


def test_single_sample_spectrogram_returns_readable_figure() -> None:
    figure = plot_spectrogram(np.array([0.0], dtype=np.float32), RATE)
    assert figure.axes


def test_dropout_repair_honors_minimum_duration() -> None:
    signal = _tone(170.0)
    corrupted = signal.copy()
    corrupted[3000:3010] = 0.0
    restored = repair_dropouts(
        corrupted,
        RATE,
        near_zero_amplitude=1e-4,
        minimum_duration_ms=20.0,
        maximum_duration_ms=100.0,
    )
    np.testing.assert_array_equal(restored, corrupted)


def test_quality_gate_uses_configured_corruption_reduction_metrics() -> None:
    time = np.arange(RATE * 2) / RATE
    corrupted = (
        0.15 * np.sin(2 * np.pi * 50 * time)
        + 0.10 * np.sin(2 * np.pi * 100 * time)
        + 0.20 * np.sin(2 * np.pi * 300 * time)
    ).astype(np.float32)
    before = detect_corruption(corrupted, RATE)
    restored = notch_hum(
        corrupted,
        RATE,
        mains_hz=50.0,
        harmonic_count=4,
        quality_factor=30.0,
    )
    validation = validate_restoration(
        corrupted,
        restored,
        RATE,
        before_report=before,
    )
    assert validation.accepted
    assert not validation.reasons
    assert all(check.passed for check in validation.checks)


def test_harmonic_sound_is_not_detected_as_mains_hum() -> None:
    time = np.arange(RATE) / RATE
    # 100 Hz harmonic sound with multiples at 200 Hz, 300 Hz
    harmonic_signal = (
        0.5 * np.sin(2 * np.pi * 100 * time)
        + 0.3 * np.sin(2 * np.pi * 200 * time)
        + 0.2 * np.sin(2 * np.pi * 300 * time)
    ).astype(np.float32)
    report = detect_corruption(harmonic_signal, RATE)
    assert "hum" not in report.detected_kinds
    assert report.metrics["hum_prominence_db"] is None


def test_harmonic_sound_pipeline_decision_is_not_unrecoverable() -> None:
    time = np.arange(round(RATE * 1.5)) / RATE
    harmonic_signal = (
        0.4 * np.sin(2 * np.pi * 100 * time)
        + 0.25 * np.sin(2 * np.pi * 200 * time)
        + 0.15 * np.sin(2 * np.pi * 300 * time)
    ).astype(np.float32)
    pipeline = SignalGuardPipeline()
    result = pipeline.analyze_samples(harmonic_signal, RATE)
    assert result.decision is not PipelineDecision.REJECT_UNRECOVERABLE
    assert result.decision in (PipelineDecision.PASS, PipelineDecision.PASS_RESTORED)


def test_bypassed_restoration_on_mild_corruption_yields_pass() -> None:
    # A natural speech sample with mild background noise (low/moderate severity)
    speech = generate_clean_speech(f0_base=120.0, seed=100)
    rate = RATE
    rng = np.random.default_rng(42)
    sample = (speech.astype(np.float32) + (0.020 * rng.normal(0, 1, size=speech.shape)).astype(np.float32))
    pipeline = SignalGuardPipeline()
    result = pipeline.analyze_samples(sample, rate)
    # Since corruption was not severe, if restoration does not pass quality gate,
    # pipeline safely rolls back and passes the original audio rather than REJECT_UNRECOVERABLE
    assert result.decision in (PipelineDecision.PASS, PipelineDecision.PASS_RESTORED)
    assert result.decision is not PipelineDecision.REJECT_UNRECOVERABLE
