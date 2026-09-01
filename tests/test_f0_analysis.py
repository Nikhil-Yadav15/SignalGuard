"""Generated-signal tests for deterministic F0 analysis."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.signal import chirp

from src.detectors.f0_analysis import (
    F0AnalysisStatus,
    F0Settings,
    analyze_f0,
    calculate_f0_differences,
)
from src.utils import ConfigError, load_config


SAMPLE_RATE = 16_000


@pytest.fixture(scope="module")
def settings() -> F0Settings:
    return F0Settings.from_config()


def _tone(
    frequency_hz: float,
    *,
    seconds: float = 1.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    times = np.arange(round(SAMPLE_RATE * seconds), dtype=np.float64) / SAMPLE_RATE
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * times)


def _rules_by_code(result: object) -> dict[str, object]:
    return {
        evaluation.code: evaluation
        for evaluation in result.rule_evaluations  # type: ignore[attr-defined]
    }


def test_200_hz_tone_returns_stable_pitch_metrics(
    settings: F0Settings,
) -> None:
    result = analyze_f0(_tone(200.0), SAMPLE_RATE, settings=settings)
    metrics = result.metrics

    assert result.status is F0AnalysisStatus.OK
    assert result.variation_is_reliable
    assert metrics.voiced_frame_ratio >= 0.95
    assert metrics.mean_f0_hz == pytest.approx(200.0, abs=2.0)
    assert metrics.std_f0_hz is not None and metrics.std_f0_hz < 2.0
    assert metrics.f0_range_hz is not None and metrics.f0_range_hz < 5.0
    assert (
        metrics.mean_abs_frame_difference_hz is not None
        and metrics.mean_abs_frame_difference_hz < 1.0
    )
    assert (
        metrics.mean_abs_second_order_change_hz is not None
        and metrics.mean_abs_second_order_change_hz < 1.0
    )
    assert metrics.frame_difference_count == metrics.total_frame_count - 1
    assert metrics.second_order_change_count == metrics.total_frame_count - 2

    rules = _rules_by_code(result)
    assert rules["LOW_F0_STD"].triggered is True
    assert rules["LOW_F0_RANGE"].triggered is True
    assert rules["LOW_F0_FRAME_DIFFERENCE"].triggered is True
    assert rules["LOW_F0_SECOND_ORDER_CHANGE"].triggered is True


@pytest.mark.parametrize("frequency_hz", [80.0, 120.0, 200.0, 300.0, 440.0])
def test_multiple_tone_frequencies_are_estimated_accurately(
    frequency_hz: float,
    settings: F0Settings,
) -> None:
    result = analyze_f0(_tone(frequency_hz), SAMPLE_RATE, settings=settings)

    tolerance_hz = max(2.0, frequency_hz * 0.01)
    assert result.status is F0AnalysisStatus.OK
    assert result.metrics.mean_f0_hz == pytest.approx(
        frequency_hz,
        abs=tolerance_hz,
    )
    assert result.metrics.voiced_frame_ratio >= 0.90


def test_linear_chirp_produces_large_f0_variation(
    settings: F0Settings,
) -> None:
    times = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    signal = 0.5 * chirp(
        times,
        f0=120.0,
        f1=320.0,
        t1=1.0,
        method="linear",
    )

    result = analyze_f0(signal, SAMPLE_RATE, settings=settings)

    assert result.status is F0AnalysisStatus.OK
    assert result.metrics.std_f0_hz is not None
    assert result.metrics.std_f0_hz > 40.0
    assert result.metrics.f0_range_hz is not None
    assert result.metrics.f0_range_hz > 150.0
    assert result.metrics.mean_abs_frame_difference_hz is not None
    assert result.metrics.mean_abs_frame_difference_hz > 1.0
    rules = _rules_by_code(result)
    assert rules["LOW_F0_STD"].triggered is False
    assert rules["LOW_F0_RANGE"].triggered is False


def test_unvoiced_gap_reduces_ratio_without_bridging_differences(
    settings: F0Settings,
) -> None:
    signal = np.concatenate(
        [_tone(200.0, seconds=0.5), np.zeros(SAMPLE_RATE // 2), _tone(200.0, seconds=0.5)]
    )

    result = analyze_f0(signal, SAMPLE_RATE, settings=settings)

    assert result.status is F0AnalysisStatus.OK
    assert 0.60 <= result.metrics.voiced_frame_ratio <= 0.82
    assert result.metrics.mean_f0_hz == pytest.approx(200.0, abs=2.0)
    for index, difference in enumerate(
        result.track.frame_to_frame_differences_hz
    ):
        if index == 0 or not (
            result.track.voiced_mask[index]
            and result.track.voiced_mask[index - 1]
        ):
            assert difference is None


def test_silence_returns_no_false_f0(settings: F0Settings) -> None:
    result = analyze_f0(np.zeros(SAMPLE_RATE), SAMPLE_RATE, settings=settings)

    assert result.status is F0AnalysisStatus.SILENCE
    assert result.metrics.voiced_frame_count == 0
    assert result.metrics.voiced_frame_ratio == 0.0
    assert result.metrics.mean_f0_hz is None
    assert result.metrics.std_f0_hz is None
    assert result.metrics.f0_range_hz is None
    assert result.track.voiced_f0_hz == ()
    assert all(rule.triggered is None for rule in result.rule_evaluations)
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("sample_count", [0, 100, 2_047])
def test_short_audio_returns_explicit_insufficient_status(
    sample_count: int,
    settings: F0Settings,
) -> None:
    short_signal = _tone(200.0)[:sample_count]

    result = analyze_f0(short_signal, SAMPLE_RATE, settings=settings)

    assert result.status is F0AnalysisStatus.INSUFFICIENT_AUDIO
    assert result.metrics.total_frame_count == 0
    assert result.metrics.mean_f0_hz is None
    assert not result.variation_is_reliable
    assert all(rule.triggered is None for rule in result.rule_evaluations)


def test_exactly_one_frame_is_not_treated_as_reliable_variation(
    settings: F0Settings,
) -> None:
    signal = _tone(200.0)[: settings.frame_length_samples]

    result = analyze_f0(signal, SAMPLE_RATE, settings=settings)

    assert result.status is F0AnalysisStatus.INSUFFICIENT_VOICED_EVIDENCE
    assert result.metrics.total_frame_count == 1
    assert result.metrics.voiced_frame_count == 1
    assert result.metrics.mean_f0_hz == pytest.approx(200.0, abs=2.0)
    assert result.metrics.std_f0_hz is None
    assert result.metrics.f0_range_hz is None
    assert result.metrics.mean_abs_frame_difference_hz is None
    assert result.metrics.mean_abs_second_order_change_hz is None
    assert all(rule.triggered is None for rule in result.rule_evaluations[1:])


def test_seeded_white_noise_fails_periodicity_voicing_gate(
    settings: F0Settings,
) -> None:
    noise = np.random.default_rng(0).normal(0.0, 0.1, SAMPLE_RATE)

    result = analyze_f0(noise, SAMPLE_RATE, settings=settings)

    assert result.status is F0AnalysisStatus.NO_VOICED_FRAMES
    assert result.metrics.voiced_frame_ratio <= 0.20
    assert result.metrics.mean_f0_hz is None
    assert max(result.track.periodicity, default=0.0) < settings.minimum_periodicity


def test_20_db_noisy_tone_remains_accurate(settings: F0Settings) -> None:
    clean = _tone(200.0)
    rng = np.random.default_rng(7)
    noise = rng.normal(size=clean.size)
    clean_rms = np.sqrt(np.mean(clean**2))
    noise *= clean_rms / (10.0 ** (20.0 / 20.0) * np.sqrt(np.mean(noise**2)))

    result = analyze_f0(clean + noise, SAMPLE_RATE, settings=settings)

    assert result.status is F0AnalysisStatus.OK
    assert result.metrics.mean_f0_hz == pytest.approx(200.0, abs=3.0)
    assert result.metrics.voiced_frame_ratio >= 0.80


def test_analysis_is_deterministic_and_does_not_mutate_input(
    settings: F0Settings,
) -> None:
    rng = np.random.default_rng(11)
    source = _tone(220.0) + rng.normal(0.0, 0.005, SAMPLE_RATE)
    before = source.copy()

    first = analyze_f0(source, SAMPLE_RATE, settings=settings)
    second = analyze_f0(source, SAMPLE_RATE, settings=settings)

    assert np.array_equal(source, before)
    assert first == second


def test_track_contract_is_aligned_and_json_safe(settings: F0Settings) -> None:
    result = analyze_f0(_tone(200.0), SAMPLE_RATE, settings=settings)
    track_lengths = {
        len(result.track.frame_start_times_seconds),
        len(result.track.candidate_f0_hz),
        len(result.track.voiced_f0_hz),
        len(result.track.voiced_mask),
        len(result.track.frame_rms_dbfs),
        len(result.track.periodicity),
        len(result.track.frame_to_frame_differences_hz),
        len(result.track.second_order_changes_hz),
    }

    assert track_lengths == {result.metrics.total_frame_count}
    assert all(
        earlier < later
        for earlier, later in zip(
            result.track.frame_start_times_seconds,
            result.track.frame_start_times_seconds[1:],
        )
    )
    voiced_values = [
        value for value in result.track.voiced_f0_hz if value is not None
    ]
    assert all(settings.fmin_hz <= value <= settings.fmax_hz for value in voiced_values)
    payload = result.to_dict()
    assert "synthetic_score" not in payload
    assert "decision" not in payload
    json.dumps(payload, allow_nan=False)


def test_difference_calculation_uses_only_contiguous_voiced_frames() -> None:
    first, second = calculate_f0_differences(
        np.array([100.0, 110.0, 130.0]),
        np.array([True, True, True]),
    )
    gap_first, gap_second = calculate_f0_differences(
        np.array([100.0, 150.0, 200.0]),
        np.array([True, False, True]),
    )

    assert first == (None, 10.0, 20.0)
    assert second == (None, None, 10.0)
    assert gap_first == (None, None, None)
    assert gap_second == (None, None, None)


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.array([1.0 + 2.0j]),
        np.array([True, False]),
        np.array(["audio"]),
        np.zeros((2, 100), dtype=np.float32),
    ],
)
def test_invalid_audio_is_rejected(
    invalid: np.ndarray,
    settings: F0Settings,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        analyze_f0(invalid, SAMPLE_RATE, settings=settings)


@pytest.mark.parametrize("invalid_rate", [0, -1, 16_000.0, True])
def test_invalid_sample_rate_is_rejected(
    invalid_rate: object,
    settings: F0Settings,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        analyze_f0(_tone(200.0), invalid_rate, settings=settings)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"fmin_hz": 0.0}, "fmin_hz"),
        ({"fmax_hz": 60.0}, "greater than fmin_hz"),
        ({"hop_length_samples": 4_096}, "must not exceed"),
        ({"trough_threshold": 0.0}, "trough_threshold"),
        ({"minimum_periodicity": 1.1}, "minimum_periodicity"),
        ({"rms_reference_percentile": True}, "rms_reference_percentile"),
        ({"rms_reference_percentile": 0.0}, "rms_reference_percentile"),
        ({"center": True}, "center must be false"),
        ({"algorithm": "pyin"}, "algorithm must be 'yin'"),
    ],
)
def test_invalid_settings_are_rejected(
    changes: dict[str, object],
    message: str,
    settings: F0Settings,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        replace(settings, **changes)


def test_settings_are_checked_against_input_nyquist(settings: F0Settings) -> None:
    incompatible = replace(settings, fmax_hz=8_000.0)

    with pytest.raises(ValueError, match="Nyquist"):
        analyze_f0(_tone(200.0), SAMPLE_RATE, settings=incompatible)


def test_missing_f0_config_value_is_reported(settings: F0Settings) -> None:
    config = load_config()
    del config["forensics"]["f0"]["minimum_periodicity"]

    with pytest.raises(ConfigError, match="minimum_periodicity"):
        F0Settings.from_config(config)
