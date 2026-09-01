"""Tests for SignalGuard's deterministic preprocessing contract."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.preprocessing import (
    AudioLoadError,
    PreprocessingSettings,
    pcm_to_float,
    preprocess_audio,
    preprocess_samples,
    remove_dc_offset,
    resample_audio,
    safe_normalize,
    segment_audio,
    to_mono,
)


@pytest.fixture
def settings() -> PreprocessingSettings:
    """Return the canonical baseline settings without reading global state."""

    return PreprocessingSettings(
        target_sample_rate=16_000,
        maximum_input_sample_rate=384_000,
        maximum_resample_factor=64.0,
        maximum_polyphase_factor=2_048,
        target_peak=0.99,
        normalization_mode="peak_limit",
        max_normalization_gain_db=12.0,
        remove_dc_offset=True,
        segment_duration_seconds=3.0,
        include_partial_segment=True,
        pad_final_segment=False,
    )


def _tone(frequency_hz: float, sample_rate: int, seconds: float) -> np.ndarray:
    times = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * frequency_hz * times)


def test_pcm_to_float_converts_signed_and_unsigned_pcm() -> None:
    signed = np.array([-32768, 0, 32767], dtype=np.int16)
    unsigned = np.array([0, 128, 255], dtype=np.uint8)

    signed_float = pcm_to_float(signed)
    unsigned_float = pcm_to_float(unsigned)

    assert signed_float.dtype == np.float32
    assert signed_float == pytest.approx([-1.0, 0.0, 32767 / 32768])
    assert unsigned_float == pytest.approx([-1.0, 0.0, 127 / 128])


@pytest.mark.parametrize(
    ("samples", "channel_axis"),
    [
        (np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float64), 1),
        (np.array([[1.0, 0.5], [-1.0, 0.25]], dtype=np.float64), 0),
    ],
)
def test_to_mono_uses_explicit_channel_axis(
    samples: np.ndarray, channel_axis: int
) -> None:
    mono = to_mono(samples, channel_axis=channel_axis)

    assert mono.dtype == np.float32
    assert mono.flags.c_contiguous
    assert mono == pytest.approx([0.0, 0.375])


def test_remove_dc_offset_centers_without_mutating_input() -> None:
    original = (_tone(440.0, 16_000, 1.0) + 0.25).astype(np.float64)
    before = original.copy()

    centered = remove_dc_offset(original)

    assert np.array_equal(original, before)
    assert abs(float(np.mean(centered, dtype=np.float64))) < 1e-7
    assert centered.shape == original.shape


def test_safe_peak_limit_does_not_boost_quiet_audio() -> None:
    quiet = np.array([-0.2, 0.1], dtype=np.float32)
    loud = np.array([-1.2, 0.6], dtype=np.float32)

    quiet_result = safe_normalize(
        quiet,
        0.99,
        mode="peak_limit",
        max_gain_db=12.0,
    )
    loud_result = safe_normalize(
        loud,
        0.99,
        mode="peak_limit",
        max_gain_db=12.0,
    )

    assert quiet_result == pytest.approx(quiet)
    assert np.max(np.abs(loud_result)) == pytest.approx(0.99, abs=1e-6)
    assert loud_result[1] / loud_result[0] == pytest.approx(-0.5)


def test_peak_normalization_caps_gain_and_handles_silence() -> None:
    quiet = np.array([-0.01, 0.005], dtype=np.float32)
    silence = np.zeros(32, dtype=np.float32)

    boosted = safe_normalize(quiet, 0.99, mode="peak", max_gain_db=6.0)
    silent_result = safe_normalize(
        silence,
        0.99,
        mode="peak_limit",
        max_gain_db=12.0,
    )

    assert np.max(np.abs(boosted)) == pytest.approx(0.01 * 10 ** (6 / 20))
    assert np.array_equal(silent_result, silence)
    assert np.all(np.isfinite(silent_result))


def test_peak_normalization_reaches_target_when_gain_is_within_cap() -> None:
    source = np.array([-0.2, 0.1], dtype=np.float32)

    result = safe_normalize(source, 0.99, mode="peak", max_gain_db=20.0)

    assert np.max(np.abs(result)) == pytest.approx(0.99, abs=1e-6)


def test_peak_normalization_handles_very_large_gain_cap() -> None:
    source = np.array([-0.1, 0.05], dtype=np.float32)

    result = safe_normalize(source, 0.99, mode="peak", max_gain_db=1e300)

    assert np.max(np.abs(result)) == pytest.approx(0.99, abs=1e-6)
    assert np.all(np.isfinite(result))


def test_float_values_outside_float32_range_are_rejected() -> None:
    extreme = np.array([1e300, -1e300], dtype=np.float64)

    with pytest.raises(ValueError, match="float32 range"):
        pcm_to_float(extreme)
    with pytest.raises(ValueError, match="float32 range"):
        safe_normalize(
            extreme,
            0.99,
            mode="peak_limit",
            max_gain_db=12.0,
        )


@pytest.mark.parametrize("source_rate", [8_000, 44_100])
def test_resample_preserves_duration_and_tone_frequency(source_rate: int) -> None:
    source = _tone(440.0, source_rate, 1.0)

    result = resample_audio(source, source_rate, 16_000)
    frequencies = np.fft.rfftfreq(result.size, d=1 / 16_000)
    peak_frequency = frequencies[int(np.argmax(np.abs(np.fft.rfft(result))))]

    assert result.size == 16_000
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))
    assert peak_frequency == pytest.approx(440.0, abs=1.0)


def test_same_rate_resampling_returns_independent_copy() -> None:
    source = np.linspace(-0.5, 0.5, 101, dtype=np.float32)

    result = resample_audio(source, 16_000, 16_000)
    result[0] = 0.0

    assert source[0] == pytest.approx(-0.5)
    assert result.size == source.size


@pytest.mark.parametrize(
    ("source_rate", "target_rate", "message"),
    [
        (10_000_000, 16_000, "configured limit"),
        (100, 16_000, "resampling factor"),
        (16_000, 10_000_000, "configured limit"),
    ],
)
def test_resample_rejects_unsafe_rates(
    source_rate: int, target_rate: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resample_audio(np.ones(16, dtype=np.float32), source_rate, target_rate)


def test_resample_rejects_large_coprime_polyphase_factors() -> None:
    with pytest.raises(ValueError, match="polyphase factor"):
        resample_audio(np.ones(16, dtype=np.float32), 383_999, 16_000)


def test_preprocess_samples_enforces_canonical_contract(
    settings: PreprocessingSettings,
) -> None:
    source_rate = 8_000
    left = 0.4 * _tone(220.0, source_rate, 1.25) + 0.20
    right = 0.3 * _tone(440.0, source_rate, 1.25) - 0.05
    stereo = np.column_stack([left, right])
    before = stereo.copy()

    result = preprocess_samples(
        stereo,
        source_rate,
        settings=settings,
        channel_axis=1,
    )

    assert np.array_equal(stereo, before)
    assert result.sample_rate == 16_000
    assert result.original_sample_rate == source_rate
    assert result.original_channels == 2
    assert result.samples.shape == (20_000,)
    assert result.samples.dtype == np.float32
    assert result.samples.flags.c_contiguous
    assert np.all(np.isfinite(result.samples))
    assert abs(float(np.mean(result.samples, dtype=np.float64))) < 1e-7
    assert np.max(np.abs(result.samples)) <= 0.99 + 1e-6
    assert result.applied_gain <= 1.0
    assert float(np.sqrt(np.mean(result.samples.astype(np.float64) ** 2))) > 0.05
    frequencies = np.fft.rfftfreq(result.samples.size, d=1 / result.sample_rate)
    spectrum = np.abs(np.fft.rfft(result.samples))
    strongest = frequencies[np.argpartition(spectrum, -4)[-4:]]
    assert np.min(np.abs(strongest - 220.0)) <= 1.0


def test_preprocess_constant_signal_becomes_finite_silence(
    settings: PreprocessingSettings,
) -> None:
    result = preprocess_samples(
        np.full(800, 0.25, dtype=np.float32),
        8_000,
        settings=settings,
    )

    assert np.all(np.isfinite(result.samples))
    assert np.max(np.abs(result.samples)) < 1e-6


def test_preprocess_can_preserve_dc_when_configured(
    settings: PreprocessingSettings,
) -> None:
    keep_dc = replace(settings, remove_dc_offset=False)

    result = preprocess_samples(
        np.full(1_000, 0.25, dtype=np.float32),
        16_000,
        settings=keep_dc,
    )

    assert np.mean(result.samples) == pytest.approx(0.25)
    assert result.removed_dc == 0.0


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([], dtype=np.float32),
        np.array([0.0, np.nan], dtype=np.float32),
        np.array([0.0, np.inf], dtype=np.float32),
        np.array([1.0 + 2.0j]),
        np.zeros((2, 2, 2), dtype=np.float32),
    ],
)
def test_preprocess_rejects_invalid_audio(
    invalid: np.ndarray,
    settings: PreprocessingSettings,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        preprocess_samples(invalid, 16_000, settings=settings)


@pytest.mark.parametrize("sample_rate", [0, -1, 16_000.0, True])
def test_resample_rejects_invalid_sample_rates(sample_rate: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        resample_audio(np.ones(16, dtype=np.float32), sample_rate, 16_000)  # type: ignore[arg-type]


def test_segmentation_is_lossless_and_tracks_partial_tail(
    settings: PreprocessingSettings,
) -> None:
    source = np.linspace(-0.5, 0.5, 100_000, dtype=np.float32)

    segments = segment_audio(source, 16_000, settings=settings)

    assert [segment.index for segment in segments] == [0, 1, 2]
    assert [segment.samples.size for segment in segments] == [48_000, 48_000, 4_000]
    assert [(segment.start_sample, segment.end_sample) for segment in segments] == [
        (0, 48_000),
        (48_000, 96_000),
        (96_000, 100_000),
    ]
    assert not any(segment.is_padded for segment in segments)
    assert np.array_equal(np.concatenate([item.samples for item in segments]), source)


def test_segmentation_handles_exact_short_empty_and_padded_inputs(
    settings: PreprocessingSettings,
) -> None:
    exact = segment_audio(np.zeros(96_000), 16_000, settings=settings)
    short = segment_audio(np.ones(100), 16_000, settings=settings)
    empty = segment_audio(np.array([], dtype=np.float32), 16_000, settings=settings)
    padded = segment_audio(
        np.ones(100),
        16_000,
        settings=settings,
        pad_final=True,
    )

    assert [segment.samples.size for segment in exact] == [48_000, 48_000]
    assert len(short) == 1 and short[0].valid_sample_count == 100
    assert empty == []
    assert padded[0].samples.size == 48_000
    assert padded[0].valid_sample_count == 100
    assert padded[0].is_padded
    assert np.array_equal(padded[0].samples[:100], np.ones(100, dtype=np.float32))
    assert np.all(padded[0].samples[100:] == 0.0)


def test_segment_can_drop_partial_tail(settings: PreprocessingSettings) -> None:
    source = np.arange(50_000, dtype=np.float32)

    segments = segment_audio(
        source,
        16_000,
        settings=settings,
        include_partial=False,
        pad_final=False,
    )

    assert len(segments) == 1
    assert segments[0].samples.size == 48_000
    assert segments[0].start_sample == 0
    assert segments[0].end_sample == 48_000
    assert np.array_equal(segments[0].samples, source[:48_000])


def test_empty_segmentation_still_validates_options(
    settings: PreprocessingSettings,
) -> None:
    empty = np.array([], dtype=np.float32)

    with pytest.raises(ValueError, match="greater than zero"):
        segment_audio(
            empty,
            16_000,
            settings=settings,
            duration_seconds=0.0,
        )
    with pytest.raises(ValueError, match="requires include_partial"):
        segment_audio(
            empty,
            16_000,
            settings=settings,
            include_partial=False,
            pad_final=True,
        )


def test_preprocess_wav_round_trip_is_deterministic(
    tmp_path: Path,
    settings: PreprocessingSettings,
) -> None:
    source_rate = 8_000
    left = 0.3 * _tone(220.0, source_rate, 1.25) + 0.10
    right = 0.2 * _tone(330.0, source_rate, 1.25) - 0.05
    audio_path = tmp_path / "stereo.wav"
    sf.write(audio_path, np.column_stack([left, right]), source_rate, subtype="PCM_16")

    first = preprocess_audio(audio_path, settings=settings)
    second = preprocess_audio(audio_path, settings=settings)

    assert first.sample_rate == 16_000
    assert first.original_channels == 2
    assert first.samples.size == 20_000
    assert abs(float(np.mean(first.samples, dtype=np.float64))) < 1e-7
    assert np.max(np.abs(first.samples)) <= 0.99 + 1e-6
    assert float(np.sqrt(np.mean(first.samples.astype(np.float64) ** 2))) > 0.05
    assert np.array_equal(first.samples, second.samples)
    assert first.applied_gain == second.applied_gain
    assert first.removed_dc == second.removed_dc


def test_preprocess_silent_wav_is_safe(
    tmp_path: Path,
    settings: PreprocessingSettings,
) -> None:
    audio_path = tmp_path / "silence.wav"
    sf.write(audio_path, np.zeros(400, dtype=np.float32), 8_000, subtype="FLOAT")

    result = preprocess_audio(audio_path, settings=settings)

    assert result.samples.size == 800
    assert np.array_equal(result.samples, np.zeros(800, dtype=np.float32))


def test_seekable_audio_stream_is_reusable_and_position_is_restored(
    settings: PreprocessingSettings,
) -> None:
    stream = BytesIO()
    sf.write(
        stream,
        0.25 * _tone(440.0, 8_000, 0.25),
        8_000,
        format="WAV",
        subtype="PCM_16",
    )
    original_position = stream.tell()

    first = preprocess_audio(stream, settings=settings)
    assert stream.tell() == original_position
    second = preprocess_audio(stream, settings=settings)

    assert stream.tell() == original_position
    assert np.array_equal(first.samples, second.samples)
    assert np.max(np.abs(first.samples)) > 0.1


def test_audio_loader_has_stable_errors(
    tmp_path: Path,
    settings: PreprocessingSettings,
) -> None:
    with pytest.raises(FileNotFoundError):
        preprocess_audio(tmp_path / "missing.wav", settings=settings)

    corrupt_path = tmp_path / "corrupt.wav"
    corrupt_path.write_bytes(b"this is not audio")
    with pytest.raises(AudioLoadError, match="Unable to decode"):
        preprocess_audio(corrupt_path, settings=settings)

    with pytest.raises(AudioLoadError, match="not a file"):
        preprocess_audio(tmp_path, settings=settings)
