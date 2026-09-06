"""Unit tests for the new LPC, bispectrum, and modulation forensic detectors."""

from __future__ import annotations

import json

import numpy as np

from src.detectors import analyze_bispectrum, analyze_lpc, analyze_modulation
from src.pipeline import PipelineDecision, SignalGuardPipeline
from src.scoring import EvidenceDecision, score_synthetic_evidence

RATE = 16_000


def tone(frequency: float, seconds: float = 1.0) -> np.ndarray:
    time = np.arange(round(RATE * seconds)) / RATE
    return (0.4 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def multi_harmonic_signal(f0: float = 150.0, seconds: float = 1.0) -> np.ndarray:
    time = np.arange(round(RATE * seconds)) / RATE
    sig = (
        0.3 * np.sin(2 * np.pi * f0 * time)
        + 0.2 * np.sin(2 * np.pi * 2 * f0 * time)
        + 0.1 * np.sin(2 * np.pi * 3 * f0 * time)
    )
    return sig.astype(np.float32)


def am_modulated_signal(f_carrier: float = 300.0, f_mod: float = 4.0, seconds: float = 1.5) -> np.ndarray:
    time = np.arange(round(RATE * seconds)) / RATE
    carrier = np.sin(2 * np.pi * f_carrier * time)
    modulator = 0.5 * (1.0 + np.sin(2 * np.pi * f_mod * time))
    return (0.4 * carrier * modulator).astype(np.float32)


class TestLpcAnalysis:
    def test_lpc_extracts_finite_metrics_on_harmonic_signal(self) -> None:
        signal = multi_harmonic_signal(200.0, seconds=1.0)
        result = analyze_lpc(signal, RATE)
        assert result.status == "OK"
        assert result.metrics["residual_kurtosis"] is not None
        assert np.isfinite(float(result.metrics["residual_kurtosis"]))
        assert result.metrics["prediction_gain_db"] is not None
        assert result.metrics["residual_skewness"] is not None
        assert len(result.rule_evaluations) == 2
        for rule in result.rule_evaluations:
            assert rule.observed_value is None or np.isfinite(rule.observed_value)

    def test_lpc_handles_silence(self) -> None:
        silence = np.zeros(RATE, dtype=np.float32)
        result = analyze_lpc(silence, RATE)
        assert result.status == "SILENCE"
        assert len(result.rule_evaluations) == 0

    def test_lpc_handles_insufficient_audio(self) -> None:
        short = np.array([0.1, -0.1, 0.2], dtype=np.float32)
        result = analyze_lpc(short, RATE)
        assert result.status == "INSUFFICIENT_AUDIO"


class TestBispectrumAnalysis:
    def test_bispectrum_extracts_bicoherence(self) -> None:
        signal = multi_harmonic_signal(200.0, seconds=1.0)
        result = analyze_bispectrum(signal, RATE)
        assert result.status == "OK"
        mean_bic = float(result.metrics["mean_bicoherence"])
        max_bic = float(result.metrics["max_bicoherence"])
        assert 0.0 <= mean_bic <= 1.0
        assert 0.0 <= max_bic <= 1.0
        assert len(result.rule_evaluations) == 2

    def test_bispectrum_handles_silence(self) -> None:
        silence = np.zeros(RATE, dtype=np.float32)
        result = analyze_bispectrum(silence, RATE)
        assert result.status == "SILENCE"

    def test_bispectrum_handles_insufficient_audio(self) -> None:
        short = np.array([0.1, 0.2], dtype=np.float32)
        result = analyze_bispectrum(short, RATE)
        assert result.status == "INSUFFICIENT_AUDIO"


class TestModulationAnalysis:
    def test_modulation_extracts_syllabic_metrics(self) -> None:
        signal = am_modulated_signal(300.0, 4.0, seconds=1.5)
        result = analyze_modulation(signal, RATE)
        assert result.status == "OK"
        assert result.metrics["syllabic_energy_ratio"] is not None
        assert 0.0 <= float(result.metrics["syllabic_energy_ratio"]) <= 1.0
        assert result.metrics["modulation_depth"] is not None
        assert len(result.rule_evaluations) == 2

    def test_modulation_handles_silence(self) -> None:
        silence = np.zeros(RATE, dtype=np.float32)
        result = analyze_modulation(silence, RATE)
        assert result.status == "SILENCE"

    def test_modulation_handles_insufficient_audio(self) -> None:
        short = np.ones(100, dtype=np.float32)
        result = analyze_modulation(short, RATE)
        assert result.status == "INSUFFICIENT_AUDIO"


class TestPipelineIntegration:
    def test_pipeline_includes_all_forensic_channels(self) -> None:
        signal = multi_harmonic_signal(180.0, seconds=1.0)
        pipeline = SignalGuardPipeline()
        result = pipeline.analyze_samples(signal, RATE)

        # Check all channels present in forensic_results
        expected_channels = {"f0", "harmonic", "spectral", "phase", "temporal", "lpc", "bispectrum", "modulation"}
        assert expected_channels.issubset(result.forensic_results.keys())

        # Check evidence scoring works
        assert 0.0 <= result.evidence.score <= 100.0
        assert result.evidence.decision in set(EvidenceDecision)
        assert result.decision in set(PipelineDecision)

        # Strict JSON serialization test
        data = result.to_dict()
        serialized = json.dumps(data, allow_nan=False)
        assert '"lpc"' in serialized
        assert '"bispectrum"' in serialized
        assert '"modulation"' in serialized

        # Check score_synthetic_evidence directly
        direct = score_synthetic_evidence(result.forensic_results)
        assert 0.0 <= direct.score <= 100.0
