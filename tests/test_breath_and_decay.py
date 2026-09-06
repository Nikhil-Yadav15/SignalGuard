"""Unit tests for breath/pause dynamics and boundary decay forensic detectors."""

from __future__ import annotations

import json

import numpy as np

from src.detectors import analyze_breath, analyze_decay
from src.pipeline import PipelineDecision, SignalGuardPipeline
from src.scoring import EvidenceDecision

RATE = 16_000


def tone(frequency: float, seconds: float = 1.0) -> np.ndarray:
    time = np.arange(round(RATE * seconds)) / RATE
    return (0.4 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


class TestBreathAnalysis:
    def test_breath_handles_silence(self) -> None:
        silence = np.zeros(RATE, dtype=np.float32)
        result = analyze_breath(silence, RATE)
        assert result.status == "SILENCE"
        assert len(result.rule_evaluations) == 0

    def test_breath_handles_insufficient_audio(self) -> None:
        short = np.array([0.1, -0.1, 0.2], dtype=np.float32)
        result = analyze_breath(short, RATE)
        assert result.status == "INSUFFICIENT_AUDIO"

    def test_breath_detects_digital_dead_silence_in_gaps(self) -> None:
        # Create speech tone with 500 ms of exact mathematical zeros
        part1 = tone(200.0, 1.0)
        gap = np.zeros(int(RATE * 0.5), dtype=np.float32)  # absolute digital silence
        part2 = tone(200.0, 1.0)
        signal = np.concatenate([part1, gap, part2])

        result = analyze_breath(signal, RATE)
        assert result.status == "OK"
        assert result.metrics["digital_silence_ratio"] is not None
        assert float(result.metrics["digital_silence_ratio"]) > 0.50
        # Check rule triggered
        silence_rules = [r for r in result.rule_evaluations if r.code == "digital_silence_pauses"]
        assert len(silence_rules) == 1
        assert silence_rules[0].triggered is True

    def test_breath_detects_run_on_speech(self) -> None:
        # 5.5s continuous speech without any pause
        long_speech = tone(220.0, 5.5)
        result = analyze_breath(long_speech, RATE)
        assert result.status == "OK"
        assert float(result.metrics["max_speech_burst_seconds"]) >= 5.0
        run_on_rules = [r for r in result.rule_evaluations if r.code == "run_on_speech"]
        assert len(run_on_rules) == 1
        assert run_on_rules[0].triggered is True


class TestDecayAnalysis:
    def test_decay_handles_silence(self) -> None:
        silence = np.zeros(RATE, dtype=np.float32)
        result = analyze_decay(silence, RATE)
        assert result.status == "SILENCE"

    def test_decay_handles_insufficient_audio(self) -> None:
        short = np.array([0.1, -0.1], dtype=np.float32)
        result = analyze_decay(short, RATE)
        assert result.status == "INSUFFICIENT_AUDIO"

    def test_decay_detects_abrupt_cutoff(self) -> None:
        # Speech tone that terminates instantly with a rectangular step into silence
        speech = tone(300.0, 0.5)
        silence = np.zeros(int(RATE * 0.3), dtype=np.float32)
        abrupt_signal = np.concatenate([speech, silence])

        result = analyze_decay(abrupt_signal, RATE)
        assert result.status == "OK"
        min_decay = float(result.metrics["min_offset_decay_ms"])
        assert min_decay < 8.0
        cutoff_rules = [r for r in result.rule_evaluations if r.code == "abrupt_utterance_cutoff"]
        assert len(cutoff_rules) == 1
        assert cutoff_rules[0].triggered is True

    def test_decay_measures_natural_exponential_decay(self) -> None:
        # Speech tone with a smooth exponential reverberation tail (tau = 40 ms)
        t_tail = np.linspace(0, 0.20, int(RATE * 0.20), endpoint=False)
        tail = (0.4 * np.sin(2 * np.pi * 300.0 * t_tail) * np.exp(-t_tail / 0.040)).astype(np.float32)
        body = tone(300.0, 0.4)
        silence = np.zeros(int(RATE * 0.2), dtype=np.float32)
        natural_signal = np.concatenate([body, tail, silence])

        result = analyze_decay(natural_signal, RATE)
        assert result.status == "OK"
        min_decay = float(result.metrics["min_offset_decay_ms"])
        assert min_decay > 8.0


class TestPipelineBreathAndDecayIntegration:
    def test_pipeline_includes_breath_and_decay(self) -> None:
        signal = tone(200.0, 1.0)
        pipeline = SignalGuardPipeline()
        result = pipeline.analyze_samples(signal, RATE)

        # Check breath and decay channels in forensic_results
        assert "breath" in result.forensic_results
        assert "decay" in result.forensic_results

        # Verify pipeline execution and decisions
        assert 0.0 <= result.evidence.score <= 100.0
        assert result.evidence.decision in set(EvidenceDecision)
        assert result.decision in set(PipelineDecision)

        # Strict JSON serialization test
        serialized = json.dumps(result.to_dict(), allow_nan=False)
        assert '"breath"' in serialized
        assert '"decay"' in serialized
