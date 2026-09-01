"""Generated-signal tests for the remaining deterministic SignalGuard SDK."""

from __future__ import annotations

import numpy as np

from src.detectors import analyze_harmonics, analyze_phase, analyze_spectral, analyze_temporal, detect_corruption
from src.pipeline import PipelineDecision, SignalGuardPipeline
from src.restoration import notch_hum, repair_dropouts, repair_impulses
from src.scoring import EvidenceDecision, score_synthetic_evidence
from src.visualization import plot_before_after, plot_spectrogram, plot_waveform


RATE = 16_000


def tone(frequency: float, seconds: float = 3.0) -> np.ndarray:
    time = np.arange(round(RATE * seconds)) / RATE
    return (0.4 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_harmonic_spectral_and_phase_results_are_finite_and_explainable() -> None:
    signal = tone(200.0)
    harmonic = analyze_harmonics(signal, RATE)
    spectral = analyze_spectral(signal, RATE)
    phase = analyze_phase(signal, RATE)
    assert harmonic.status == "OK"
    assert harmonic.metrics["harmonic_energy_ratio"] is not None
    assert harmonic.metrics["harmonic_energy_ratio"] > 0.80
    assert spectral.status == "OK" and spectral.metrics["frame_count"] > 1
    assert phase.status == "OK" and 0 <= phase.metrics["phase_jump_ratio"] <= 1
    assert all(rule.observed_value is None or np.isfinite(rule.observed_value) for result in (harmonic, spectral, phase) for rule in result.rule_evaluations)


def test_temporal_analysis_exposes_cross_segment_variability() -> None:
    result = analyze_temporal([tone(140.0), tone(280.0)], RATE)
    assert result.status == "OK"
    assert result.metrics["segment_count"] == 2
    assert result.metrics["centroid_cv"] is not None and result.metrics["centroid_cv"] > 0


def test_hum_and_impulse_detection_and_targeted_repair() -> None:
    time = np.arange(RATE * 2) / RATE
    hum = (0.15 * np.sin(2 * np.pi * 50 * time) + 0.10 * np.sin(2 * np.pi * 100 * time)).astype(np.float32)
    report = detect_corruption(hum, RATE)
    assert "hum" in report.detected_kinds
    restored = notch_hum(hum, RATE, mains_hz=50, harmonic_count=4, quality_factor=30)
    before = float(np.abs(np.fft.rfft(hum))[round(50 * hum.size / RATE)])
    after = float(np.abs(np.fft.rfft(restored))[round(50 * hum.size / RATE)])
    assert after < before * 0.30
    impulses = tone(190.0); impulses[2000] = 1.0; impulses[9000] = -1.0
    assert "impulse" in detect_corruption(impulses, RATE).detected_kinds
    repaired = repair_impulses(impulses, RATE, kernel_size=5, context_samples=4)
    assert abs(repaired[2000]) < abs(impulses[2000])


def test_dropout_repair_preserves_length_and_interpolates_short_gap() -> None:
    corrupted = tone(170.0); corrupted[5000:5500] = 0
    restored = repair_dropouts(corrupted, RATE, near_zero_amplitude=1e-4, maximum_duration_ms=100)
    assert restored.shape == corrupted.shape
    assert np.count_nonzero(restored[5005:5495]) > 0


def test_scoring_and_pipeline_return_structured_decisions() -> None:
    signal = tone(200.0)
    pipeline = SignalGuardPipeline()
    result = pipeline.analyze_samples(signal, RATE)
    assert result.decision in set(PipelineDecision)
    assert 0 <= result.evidence.score <= 100
    assert result.evidence.decision in set(EvidenceDecision)
    assert result.to_dict()["decision"] == result.decision.value
    direct = score_synthetic_evidence(result.forensic_results)
    assert direct.score == result.evidence.score


def test_visualizations_return_figures_without_showing_gui() -> None:
    signal = tone(200.0)
    figures = (plot_waveform(signal, RATE), plot_spectrogram(signal, RATE), plot_before_after(signal, signal, RATE))
    assert all(figure.axes for figure in figures)
    for figure in figures:
        figure.canvas.draw()
