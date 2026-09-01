"""Deterministic canonicalization and segmentation for speech audio."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import gcd
from numbers import Real
from pathlib import Path
from typing import Any, BinaryIO, Literal, TypeAlias

import numpy as np
import soundfile as sf
from numpy.typing import ArrayLike, NDArray
from scipy.signal import resample_poly

from .utils import (
    ConfigError,
    get_config_section,
    load_config,
    seconds_to_samples,
    validate_audio_array,
    validate_sample_rate,
)


NormalizationMode: TypeAlias = Literal["peak_limit", "peak"]
AudioSource: TypeAlias = str | Path | BinaryIO


class AudioLoadError(RuntimeError):
    """Raised when an audio source cannot be decoded into samples."""


@dataclass(frozen=True, slots=True)
class PreprocessingSettings:
    """Validated preprocessing settings sourced from ``config.yaml``."""

    target_sample_rate: int
    maximum_input_sample_rate: int
    maximum_resample_factor: float
    maximum_polyphase_factor: int
    target_peak: float
    normalization_mode: NormalizationMode
    max_normalization_gain_db: float
    remove_dc_offset: bool
    segment_duration_seconds: float
    include_partial_segment: bool
    pad_final_segment: bool

    def __post_init__(self) -> None:
        maximum_rate = validate_sample_rate(
            self.maximum_input_sample_rate,
            name="maximum_input_sample_rate",
        )
        target_rate = validate_sample_rate(
            self.target_sample_rate,
            name="target_sample_rate",
        )
        if target_rate > maximum_rate:
            raise ValueError(
                "target_sample_rate must not exceed maximum_input_sample_rate"
            )
        _validate_positive_real(
            self.maximum_resample_factor,
            name="maximum_resample_factor",
        )
        if self.maximum_resample_factor < 1.0:
            raise ValueError("maximum_resample_factor must be at least 1.0")
        if (
            validate_sample_rate(
                self.maximum_polyphase_factor,
                name="maximum_polyphase_factor",
            )
            < 1
        ):
            raise ValueError("maximum_polyphase_factor must be positive")
        _validate_target_peak(self.target_peak)
        if self.normalization_mode not in ("peak_limit", "peak"):
            raise ValueError("normalization_mode must be 'peak_limit' or 'peak'")
        _validate_non_negative_real(
            self.max_normalization_gain_db,
            name="max_normalization_gain_db",
        )
        if not isinstance(self.remove_dc_offset, bool):
            raise TypeError("remove_dc_offset must be bool")
        _validate_positive_real(
            self.segment_duration_seconds,
            name="segment_duration_seconds",
        )
        if not isinstance(self.include_partial_segment, bool):
            raise TypeError("include_partial_segment must be bool")
        if not isinstance(self.pad_final_segment, bool):
            raise TypeError("pad_final_segment must be bool")
        if self.pad_final_segment and not self.include_partial_segment:
            raise ValueError(
                "pad_final_segment requires include_partial_segment to be true"
            )

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | None = None
    ) -> "PreprocessingSettings":
        """Build settings from a loaded configuration or the default YAML."""

        root = load_config() if config is None else config
        section = get_config_section(root, "preprocessing")
        required_keys = (
            "target_sample_rate",
            "maximum_input_sample_rate",
            "maximum_resample_factor",
            "maximum_polyphase_factor",
            "target_peak",
            "normalization_mode",
            "max_normalization_gain_db",
            "remove_dc_offset",
            "segment_duration_seconds",
            "include_partial_segment",
            "pad_final_segment",
        )
        missing = [key for key in required_keys if key not in section]
        if missing:
            raise ConfigError(
                "Missing preprocessing configuration values: " + ", ".join(missing)
            )
        return cls(
            target_sample_rate=section["target_sample_rate"],
            maximum_input_sample_rate=section["maximum_input_sample_rate"],
            maximum_resample_factor=section["maximum_resample_factor"],
            maximum_polyphase_factor=section["maximum_polyphase_factor"],
            target_peak=section["target_peak"],
            normalization_mode=section["normalization_mode"],
            max_normalization_gain_db=section["max_normalization_gain_db"],
            remove_dc_offset=section["remove_dc_offset"],
            segment_duration_seconds=section["segment_duration_seconds"],
            include_partial_segment=section["include_partial_segment"],
            pad_final_segment=section["pad_final_segment"],
        )


@dataclass(frozen=True, slots=True, eq=False)
class PreprocessedAudio:
    """Canonical mono audio plus provenance from preprocessing."""

    samples: NDArray[np.float32]
    sample_rate: int
    original_sample_rate: int
    original_channels: int
    removed_dc: float
    applied_gain: float

    @property
    def duration_seconds(self) -> float:
        """Duration of the canonical signal in seconds."""

        return self.samples.size / self.sample_rate


@dataclass(frozen=True, slots=True, eq=False)
class AudioSegment:
    """A non-overlapping audio segment with source sample offsets."""

    index: int
    samples: NDArray[np.float32]
    sample_rate: int
    start_sample: int
    end_sample: int

    @property
    def valid_sample_count(self) -> int:
        """Number of real source samples, excluding optional right padding."""

        return self.end_sample - self.start_sample

    @property
    def start_seconds(self) -> float:
        """Segment start time relative to the canonical clip."""

        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        """Exclusive end time of valid source audio."""

        return self.end_sample / self.sample_rate

    @property
    def is_padded(self) -> bool:
        """Whether the segment has zero-padding beyond its valid samples."""

        return self.samples.size > self.valid_sample_count


def pcm_to_float(samples: ArrayLike) -> NDArray[np.float32]:
    """Convert real PCM or floating audio to an independent ``float32`` array.

    Signed integers use their full symmetric container range (for example,
    ``int16`` divides by 32768). Unsigned PCM is centered at its midpoint before
    scaling. Floating-point amplitudes are preserved until safe normalization.
    """

    array = validate_audio_array(samples)
    if np.issubdtype(array.dtype, np.floating):
        converted = array.astype(np.float64, copy=True)
    elif np.issubdtype(array.dtype, np.signedinteger):
        limits = np.iinfo(array.dtype)
        scale = float(max(abs(limits.min), abs(limits.max)))
        converted = array.astype(np.float64) / scale
    elif np.issubdtype(array.dtype, np.unsignedinteger):
        limits = np.iinfo(array.dtype)
        midpoint = (float(limits.max) + 1.0) / 2.0
        converted = (array.astype(np.float64) - midpoint) / midpoint
    else:  # pragma: no cover - validation rejects non-real numeric dtypes.
        raise TypeError("samples must contain integer PCM or floating-point audio")
    return _as_float32_checked(converted, context="PCM conversion")


def to_mono(samples: ArrayLike, *, channel_axis: int = -1) -> NDArray[np.float32]:
    """Fold one- or two-dimensional audio down to mono by channel mean.

    The channel axis is explicit because SoundFile uses frames-first arrays while
    some audio libraries use channels-first arrays. Anti-phase channels can
    cancel under arithmetic averaging; this is expected downmix behavior.
    """

    array = pcm_to_float(samples)
    if array.ndim == 1:
        return array
    normalized_axis = _normalize_axis(channel_axis, array.ndim)
    mono = np.mean(array, axis=normalized_axis, dtype=np.float64)
    return _as_float32_checked(mono, context="mono fold-down")


def remove_dc_offset(
    samples: ArrayLike, *, sample_axis: int = 0
) -> NDArray[np.float32]:
    """Subtract the sample mean using float64 accumulation."""

    array = pcm_to_float(samples)
    normalized_axis = _normalize_axis(sample_axis, array.ndim)
    offset = np.mean(array, axis=normalized_axis, dtype=np.float64, keepdims=True)
    centered = array.astype(np.float64) - offset
    return _as_float32_checked(centered, context="DC-offset removal")


def resample_audio(
    samples: ArrayLike,
    original_sample_rate: int,
    target_sample_rate: int,
    *,
    maximum_sample_rate: int | None = None,
    maximum_resample_factor: float | None = None,
    maximum_polyphase_factor: int | None = None,
) -> NDArray[np.float32]:
    """Resample mono audio with deterministic polyphase filtering."""

    if (
        maximum_sample_rate is None
        or maximum_resample_factor is None
        or maximum_polyphase_factor is None
    ):
        default_settings = PreprocessingSettings.from_config()
        maximum_sample_rate = (
            default_settings.maximum_input_sample_rate
            if maximum_sample_rate is None
            else maximum_sample_rate
        )
        maximum_resample_factor = (
            default_settings.maximum_resample_factor
            if maximum_resample_factor is None
            else maximum_resample_factor
        )
        maximum_polyphase_factor = (
            default_settings.maximum_polyphase_factor
            if maximum_polyphase_factor is None
            else maximum_polyphase_factor
        )
    maximum_rate = validate_sample_rate(
        maximum_sample_rate,
        name="maximum_sample_rate",
    )
    _validate_positive_real(
        maximum_resample_factor,
        name="maximum_resample_factor",
    )
    if maximum_resample_factor < 1.0:
        raise ValueError("maximum_resample_factor must be at least 1.0")
    polyphase_limit = validate_sample_rate(
        maximum_polyphase_factor,
        name="maximum_polyphase_factor",
    )

    source_rate = validate_sample_rate(
        original_sample_rate, name="original_sample_rate"
    )
    destination_rate = validate_sample_rate(
        target_sample_rate, name="target_sample_rate"
    )
    if source_rate > maximum_rate or destination_rate > maximum_rate:
        raise ValueError(
            f"sample rates must not exceed the configured limit of {maximum_rate} Hz"
        )
    resample_factor = max(
        source_rate / destination_rate,
        destination_rate / source_rate,
    )
    if resample_factor > float(maximum_resample_factor):
        raise ValueError(
            "resampling factor exceeds the configured safety limit of "
            f"{maximum_resample_factor:g}"
        )
    mono = validate_audio_array(samples, allowed_dimensions=(1,))
    values = pcm_to_float(mono)
    if source_rate == destination_rate:
        return values.copy()

    common_divisor = gcd(source_rate, destination_rate)
    up = destination_rate // common_divisor
    down = source_rate // common_divisor
    if max(up, down) > polyphase_limit:
        raise ValueError(
            "reduced polyphase factor exceeds the configured safety limit of "
            f"{polyphase_limit}"
        )
    resampled = resample_poly(values.astype(np.float64), up, down)

    expected_length = max(1, int(round(values.size * destination_rate / source_rate)))
    if resampled.size > expected_length:
        resampled = resampled[:expected_length]
    elif resampled.size < expected_length:
        resampled = np.pad(
            resampled,
            (0, expected_length - resampled.size),
            mode="edge",
        )
    return _as_float32_checked(resampled, context="polyphase resampling")


def safe_normalize(
    samples: ArrayLike,
    target_peak: float,
    *,
    mode: NormalizationMode,
    max_gain_db: float,
) -> NDArray[np.float32]:
    """Safely normalize mono audio without division-by-zero on silence.

    ``peak_limit`` only attenuates signals above ``target_peak``. ``peak`` may
    also boost quiet signals, but its gain is capped by ``max_gain_db``.
    """

    normalized, _ = _normalize_with_gain(
        samples,
        target_peak=target_peak,
        mode=mode,
        max_gain_db=max_gain_db,
    )
    return normalized


def load_audio(source: AudioSource) -> tuple[NDArray[np.float32], int]:
    """Decode an audio source as a frames-by-channels float array.

    WAV support is the cross-platform baseline. Other formats, including MP3,
    depend on the codecs available in the installed libsndfile build.
    """

    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Audio source does not exist: {source_path}")
        if not source_path.is_file():
            raise AudioLoadError(f"Audio source is not a file: {source_path}")

    original_stream_position: int | None = None
    if not isinstance(source, (str, Path)):
        try:
            original_stream_position = source.tell()
            source.seek(0)
        except (AttributeError, OSError, ValueError):
            original_stream_position = None

    try:
        samples, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioLoadError("Unable to decode the supplied audio source") from exc
    finally:
        if original_stream_position is not None:
            try:
                source.seek(original_stream_position)
            except (AttributeError, OSError, ValueError):
                pass

    try:
        validate_sample_rate(sample_rate)
        validate_audio_array(samples)
    except (TypeError, ValueError) as exc:
        raise AudioLoadError("Decoded audio is empty or invalid") from exc
    return np.ascontiguousarray(samples, dtype=np.float32), int(sample_rate)


def preprocess_samples(
    samples: ArrayLike,
    sample_rate: int,
    *,
    settings: PreprocessingSettings | None = None,
    channel_axis: int = -1,
) -> PreprocessedAudio:
    """Convert in-memory audio to SignalGuard's canonical representation."""

    active_settings = settings or PreprocessingSettings.from_config()
    source_rate = validate_sample_rate(sample_rate)
    source = validate_audio_array(samples)
    original_channels = (
        1 if source.ndim == 1 else source.shape[_normalize_axis(channel_axis, 2)]
    )

    mono = to_mono(source, channel_axis=channel_axis)
    removed_dc = float(np.mean(mono, dtype=np.float64))
    if active_settings.remove_dc_offset:
        mono = remove_dc_offset(mono)
    else:
        removed_dc = 0.0

    resampled = resample_audio(
        mono,
        original_sample_rate=source_rate,
        target_sample_rate=active_settings.target_sample_rate,
        maximum_sample_rate=active_settings.maximum_input_sample_rate,
        maximum_resample_factor=active_settings.maximum_resample_factor,
        maximum_polyphase_factor=active_settings.maximum_polyphase_factor,
    )
    # Polyphase edge transients can introduce a minute residual mean. Recenter
    # once more so the canonical output satisfies the DC-free contract.
    if active_settings.remove_dc_offset:
        resampled = remove_dc_offset(resampled)

    normalized, applied_gain = _normalize_with_gain(
        resampled,
        target_peak=active_settings.target_peak,
        mode=active_settings.normalization_mode,
        max_gain_db=active_settings.max_normalization_gain_db,
    )
    return PreprocessedAudio(
        samples=np.ascontiguousarray(normalized, dtype=np.float32),
        sample_rate=active_settings.target_sample_rate,
        original_sample_rate=source_rate,
        original_channels=original_channels,
        removed_dc=removed_dc,
        applied_gain=applied_gain,
    )


def preprocess_audio(
    source: AudioSource,
    *,
    settings: PreprocessingSettings | None = None,
) -> PreprocessedAudio:
    """Decode and canonicalize an audio file or binary stream."""

    samples, sample_rate = load_audio(source)
    return preprocess_samples(
        samples,
        sample_rate,
        settings=settings,
        channel_axis=1,
    )


def segment_audio(
    samples: ArrayLike,
    sample_rate: int,
    *,
    duration_seconds: float | None = None,
    include_partial: bool | None = None,
    pad_final: bool | None = None,
    settings: PreprocessingSettings | None = None,
) -> list[AudioSegment]:
    """Split canonical mono audio into ordered, non-overlapping segments.

    The final partial segment is retained and left unpadded by default so
    artificial silence does not bias later forensic features. Segment samples
    are copies; callers cannot mutate the original clip through them.
    """

    rate = validate_sample_rate(sample_rate)
    raw = validate_audio_array(
        samples,
        allow_empty=True,
        allowed_dimensions=(1,),
    )

    active_settings = settings
    if (
        duration_seconds is None
        or include_partial is None
        or pad_final is None
    ):
        active_settings = active_settings or PreprocessingSettings.from_config()

    segment_duration = (
        active_settings.segment_duration_seconds
        if duration_seconds is None
        else duration_seconds
    )
    keep_partial = (
        active_settings.include_partial_segment
        if include_partial is None
        else include_partial
    )
    pad_partial = (
        active_settings.pad_final_segment if pad_final is None else pad_final
    )
    if not isinstance(keep_partial, bool) or not isinstance(pad_partial, bool):
        raise TypeError("include_partial and pad_final must be bool")
    if pad_partial and not keep_partial:
        raise ValueError("pad_final requires include_partial to be true")

    segment_samples = seconds_to_samples(segment_duration, rate)
    if raw.size == 0:
        return []
    mono = pcm_to_float(raw)
    segments: list[AudioSegment] = []
    for start in range(0, mono.size, segment_samples):
        end = min(start + segment_samples, mono.size)
        valid_count = end - start
        if valid_count < segment_samples and not keep_partial:
            break
        segment_values = mono[start:end].copy()
        if pad_partial and valid_count < segment_samples:
            segment_values = np.pad(
                segment_values,
                (0, segment_samples - valid_count),
                mode="constant",
            )
        segments.append(
            AudioSegment(
                index=len(segments),
                samples=np.ascontiguousarray(segment_values, dtype=np.float32),
                sample_rate=rate,
                start_sample=start,
                end_sample=end,
            )
        )
    return segments


def _normalize_with_gain(
    samples: ArrayLike,
    *,
    target_peak: float,
    mode: NormalizationMode,
    max_gain_db: float,
) -> tuple[NDArray[np.float32], float]:
    values = pcm_to_float(validate_audio_array(samples, allowed_dimensions=(1,)))
    _validate_target_peak(target_peak)
    _validate_non_negative_real(max_gain_db, name="max_gain_db")
    if mode not in ("peak_limit", "peak"):
        raise ValueError("mode must be 'peak_limit' or 'peak'")

    peak = float(np.max(np.abs(values)))
    if peak == 0.0:
        return values.copy(), 1.0

    requested_gain = float(target_peak) / peak
    if mode == "peak_limit":
        gain = min(1.0, requested_gain)
    else:
        requested_gain_db = 20.0 * np.log10(requested_gain)
        applied_gain_db = min(requested_gain_db, float(max_gain_db))
        gain = 10.0 ** (applied_gain_db / 20.0)
    normalized = values.astype(np.float64) * gain
    normalized = np.clip(normalized, -float(target_peak), float(target_peak))
    return np.ascontiguousarray(normalized, dtype=np.float32), float(gain)


def _as_float32_checked(
    values: ArrayLike, *, context: str
) -> NDArray[np.float32]:
    array = np.asarray(values)
    float32_limit = np.finfo(np.float32).max
    if np.any(~np.isfinite(array)) or np.any(np.abs(array) > float32_limit):
        raise ValueError(f"{context} produced values outside the float32 range")
    converted = np.ascontiguousarray(array, dtype=np.float32)
    if np.any(~np.isfinite(converted)):
        raise ValueError(f"{context} produced non-finite float32 samples")
    return converted


def _normalize_axis(axis: int, dimensions: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("axis must be an integer")
    if axis < -dimensions or axis >= dimensions:
        raise ValueError(f"axis {axis} is invalid for a {dimensions}-D audio array")
    return axis % dimensions


def _validate_target_peak(target_peak: float) -> None:
    _validate_positive_real(target_peak, name="target_peak")
    if float(target_peak) > 1.0:
        raise ValueError("target_peak must not exceed 1.0")


def _validate_positive_real(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive real number")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")


def _validate_non_negative_real(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-negative real number")
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
