"""Deterministic fundamental-frequency analysis using the YIN algorithm.

This module reports interpretable pitch metrics and threshold-rule evaluations.
It deliberately does not calculate a Synthetic Evidence Score or make a
natural/synthetic decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import ceil
from numbers import Integral, Real
from typing import Any

import librosa
import numpy as np
from librosa.util.exceptions import ParameterError
from numpy.typing import ArrayLike, NDArray

from ..preprocessing import pcm_to_float, remove_dc_offset
from ..utils import (
    ConfigError,
    get_config_section,
    load_config,
    validate_audio_array,
    validate_sample_rate,
    to_json_safe,
)


class F0AnalysisStatus(str, Enum):
    """Outcome states for an F0 analysis attempt."""

    OK = "OK"
    SILENCE = "SILENCE"
    INSUFFICIENT_AUDIO = "INSUFFICIENT_AUDIO"
    NO_VOICED_FRAMES = "NO_VOICED_FRAMES"
    INSUFFICIENT_VOICED_EVIDENCE = "INSUFFICIENT_VOICED_EVIDENCE"


@dataclass(frozen=True, slots=True)
class F0Settings:
    """Validated YIN, voicing, and heuristic settings from ``config.yaml``."""

    algorithm: str
    fmin_hz: float
    fmax_hz: float
    frame_length_samples: int
    hop_length_samples: int
    trough_threshold: float
    center: bool
    pad_mode: str
    postprocessing: str
    std_ddof: int
    range_method: str
    delta_policy: str
    rms_floor_dbfs: float
    relative_rms_range_db: float
    rms_reference_percentile: float
    minimum_periodicity: float
    minimum_voiced_run_frames: int
    minimum_voiced_frames_for_variation: int
    minimum_voiced_ratio_for_variation: float
    low_voiced_ratio_below: float
    low_std_hz: float
    low_range_hz: float
    low_mean_abs_delta_hz_per_frame: float
    low_mean_abs_second_delta_hz_per_frame2: float

    def __post_init__(self) -> None:
        if self.algorithm != "yin":
            raise ValueError("algorithm must be 'yin'")
        _validate_positive_real(self.fmin_hz, name="fmin_hz")
        _validate_positive_real(self.fmax_hz, name="fmax_hz")
        if self.fmax_hz <= self.fmin_hz:
            raise ValueError("fmax_hz must be greater than fmin_hz")
        _validate_positive_integer(
            self.frame_length_samples,
            name="frame_length_samples",
        )
        _validate_positive_integer(
            self.hop_length_samples,
            name="hop_length_samples",
        )
        if self.hop_length_samples > self.frame_length_samples:
            raise ValueError("hop_length_samples must not exceed frame_length_samples")
        _validate_unit_interval(
            self.trough_threshold,
            name="trough_threshold",
            exclude_zero=True,
        )
        if self.center is not False:
            raise ValueError("center must be false to avoid padded boundary evidence")
        if self.pad_mode != "constant":
            raise ValueError("pad_mode must be 'constant' for the baseline profile")
        if self.postprocessing != "none":
            raise ValueError("postprocessing must be 'none' for the baseline profile")
        if self.std_ddof != 0:
            raise ValueError("std_ddof must be 0 for population standard deviation")
        if self.range_method != "max_minus_min":
            raise ValueError("range_method must be 'max_minus_min'")
        if self.delta_policy != "contiguous_voiced_only":
            raise ValueError("delta_policy must be 'contiguous_voiced_only'")
        _validate_finite_real(self.rms_floor_dbfs, name="rms_floor_dbfs")
        if self.rms_floor_dbfs > 0:
            raise ValueError("rms_floor_dbfs must not exceed 0 dBFS")
        _validate_positive_real(
            self.relative_rms_range_db,
            name="relative_rms_range_db",
        )
        _validate_finite_real(
            self.rms_reference_percentile,
            name="rms_reference_percentile",
        )
        if not 0 < self.rms_reference_percentile <= 100:
            raise ValueError("rms_reference_percentile must be in (0, 100]")
        _validate_unit_interval(
            self.minimum_periodicity,
            name="minimum_periodicity",
        )
        _validate_positive_integer(
            self.minimum_voiced_run_frames,
            name="minimum_voiced_run_frames",
        )
        _validate_positive_integer(
            self.minimum_voiced_frames_for_variation,
            name="minimum_voiced_frames_for_variation",
        )
        if self.minimum_voiced_frames_for_variation < 3:
            raise ValueError(
                "minimum_voiced_frames_for_variation must be at least 3"
            )
        _validate_unit_interval(
            self.minimum_voiced_ratio_for_variation,
            name="minimum_voiced_ratio_for_variation",
        )
        _validate_unit_interval(
            self.low_voiced_ratio_below,
            name="low_voiced_ratio_below",
        )
        for name, value in (
            ("low_std_hz", self.low_std_hz),
            ("low_range_hz", self.low_range_hz),
            (
                "low_mean_abs_delta_hz_per_frame",
                self.low_mean_abs_delta_hz_per_frame,
            ),
            (
                "low_mean_abs_second_delta_hz_per_frame2",
                self.low_mean_abs_second_delta_hz_per_frame2,
            ),
        ):
            _validate_non_negative_real(value, name=name)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> "F0Settings":
        """Build settings from the default or supplied project configuration."""

        root = load_config() if config is None else config
        forensics = get_config_section(root, "forensics")
        section = get_config_section(forensics, "f0")
        field_names = tuple(cls.__dataclass_fields__)
        missing = [name for name in field_names if name not in section]
        if missing:
            raise ConfigError(
                "Missing F0 configuration values: " + ", ".join(missing)
            )
        return cls(**{name: section[name] for name in field_names})

    def validate_for_sample_rate(self, sample_rate: int) -> None:
        """Validate Nyquist and frame-duration constraints for a sample rate."""

        rate = validate_sample_rate(sample_rate)
        if self.fmax_hz >= rate / 2.0:
            raise ValueError("fmax_hz must be below the input sample rate's Nyquist")
        minimum_reliable_frame = ceil(2.0 * rate / self.fmin_hz)
        if self.frame_length_samples < minimum_reliable_frame:
            raise ValueError(
                "frame_length_samples must span at least two periods at fmin_hz"
            )


@dataclass(frozen=True, slots=True)
class F0Metrics:
    """Scalar F0 statistics calculated only from voiced frames."""

    total_frame_count: int
    voiced_frame_count: int
    voiced_frame_ratio: float
    mean_f0_hz: float | None
    std_f0_hz: float | None
    minimum_f0_hz: float | None
    maximum_f0_hz: float | None
    f0_range_hz: float | None
    mean_abs_frame_difference_hz: float | None
    mean_abs_second_order_change_hz: float | None
    frame_difference_count: int
    second_order_change_count: int


@dataclass(frozen=True, slots=True)
class F0Track:
    """Frame-aligned pitch candidates, voicing diagnostics, and differences."""

    frame_start_times_seconds: tuple[float, ...]
    candidate_f0_hz: tuple[float | None, ...]
    voiced_f0_hz: tuple[float | None, ...]
    voiced_mask: tuple[bool, ...]
    frame_rms_dbfs: tuple[float, ...]
    periodicity: tuple[float, ...]
    frame_to_frame_differences_hz: tuple[float | None, ...]
    second_order_changes_hz: tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class F0RuleEvaluation:
    """A transparent threshold comparison for later scoring consumption."""

    code: str
    observed_value: float | None
    threshold: float
    comparison: str
    triggered: bool | None
    explanation: str


@dataclass(frozen=True, slots=True)
class F0AnalysisResult:
    """Complete, JSON-safe result of deterministic F0 analysis."""

    status: F0AnalysisStatus
    algorithm: str
    sample_rate_hz: int
    frame_length_samples: int
    hop_length_samples: int
    variation_is_reliable: bool
    metrics: F0Metrics
    track: F0Track
    rule_evaluations: tuple[F0RuleEvaluation, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a nested JSON-safe dictionary without NumPy scalars or NaN."""
        return to_json_safe(self)


def analyze_f0(
    samples: ArrayLike,
    sample_rate: int,
    *,
    settings: F0Settings | None = None,
) -> F0AnalysisResult:
    """Analyze mono audio and return deterministic, interpretable F0 evidence.

    YIN supplies one candidate per complete frame. A separate energy and
    normalized-autocorrelation gate determines whether that candidate is voiced.
    Variation metrics never cross an unvoiced gap, and no score is calculated.
    """

    rate = validate_sample_rate(sample_rate)
    active_settings = settings or F0Settings.from_config()
    active_settings.validate_for_sample_rate(rate)
    source = validate_audio_array(
        samples,
        allow_empty=True,
        allowed_dimensions=(1,),
    )
    if source.size == 0:
        return _empty_result(
            F0AnalysisStatus.INSUFFICIENT_AUDIO,
            rate,
            active_settings,
            "Audio is empty; no F0 frames can be analyzed.",
        )

    values = pcm_to_float(source)
    centered = remove_dc_offset(values).astype(np.float64, copy=False)
    global_rms = float(np.sqrt(np.mean(np.square(centered))))
    silence_rms = 10.0 ** (active_settings.rms_floor_dbfs / 20.0)
    if global_rms <= silence_rms:
        return _empty_result(
            F0AnalysisStatus.SILENCE,
            rate,
            active_settings,
            "Signal RMS is at or below the configured silence floor.",
        )
    if centered.size < active_settings.frame_length_samples:
        return _empty_result(
            F0AnalysisStatus.INSUFFICIENT_AUDIO,
            rate,
            active_settings,
            "Audio is shorter than one complete YIN frame.",
        )

    try:
        candidates = librosa.yin(
            centered,
            fmin=active_settings.fmin_hz,
            fmax=active_settings.fmax_hz,
            sr=rate,
            frame_length=active_settings.frame_length_samples,
            hop_length=active_settings.hop_length_samples,
            trough_threshold=active_settings.trough_threshold,
            center=active_settings.center,
            pad_mode=active_settings.pad_mode,
        )
    except ParameterError as exc:
        raise ValueError(f"YIN analysis failed: {exc}") from exc

    frames = librosa.util.frame(
        centered,
        frame_length=active_settings.frame_length_samples,
        hop_length=active_settings.hop_length_samples,
    )
    if frames.shape[1] != candidates.size:
        raise RuntimeError("YIN and diagnostic frame counts do not match")

    frame_rms = np.sqrt(np.mean(np.square(frames), axis=0, dtype=np.float64))
    frame_rms_dbfs = _amplitude_to_dbfs(frame_rms)
    reference_dbfs = float(
        np.percentile(
            frame_rms_dbfs,
            active_settings.rms_reference_percentile,
        )
    )
    energy_threshold_dbfs = max(
        active_settings.rms_floor_dbfs,
        reference_dbfs - active_settings.relative_rms_range_db,
    )
    energy_mask = frame_rms_dbfs >= energy_threshold_dbfs

    periodicity = np.array(
        [
            _periodicity_at_candidate(frames[:, index], candidates[index], rate)
            for index in range(candidates.size)
        ],
        dtype=np.float64,
    )
    finite_candidates = np.isfinite(candidates)
    bounded_candidates = (
        finite_candidates
        & (candidates >= active_settings.fmin_hz)
        & (candidates <= active_settings.fmax_hz)
    )
    voiced_mask = (
        energy_mask
        & bounded_candidates
        & (periodicity >= active_settings.minimum_periodicity)
    )
    if voiced_mask.size >= active_settings.minimum_voiced_run_frames:
        voiced_mask = _remove_short_voiced_runs(
            voiced_mask,
            active_settings.minimum_voiced_run_frames,
        )

    first_differences, second_changes = calculate_f0_differences(
        candidates,
        voiced_mask,
    )
    metrics = _calculate_metrics(
        candidates,
        voiced_mask,
        first_differences,
        second_changes,
        ddof=active_settings.std_ddof,
    )
    variation_is_reliable = (
        metrics.voiced_frame_count
        >= active_settings.minimum_voiced_frames_for_variation
        and metrics.voiced_frame_ratio
        >= active_settings.minimum_voiced_ratio_for_variation
        and metrics.second_order_change_count > 0
    )
    if metrics.voiced_frame_count == 0:
        status = F0AnalysisStatus.NO_VOICED_FRAMES
        warnings = (
            "YIN returned candidates, but none passed the configured voicing gates.",
        )
    elif not variation_is_reliable:
        status = F0AnalysisStatus.INSUFFICIENT_VOICED_EVIDENCE
        warnings = (
            "Voiced content is insufficient for reliable F0-variation rules.",
        )
    else:
        status = F0AnalysisStatus.OK
        warnings = ()

    track = F0Track(
        frame_start_times_seconds=tuple(
            float(index * active_settings.hop_length_samples / rate)
            for index in range(candidates.size)
        ),
        candidate_f0_hz=tuple(
            float(value) if np.isfinite(value) else None for value in candidates
        ),
        voiced_f0_hz=tuple(
            float(value) if voiced else None
            for value, voiced in zip(candidates, voiced_mask)
        ),
        voiced_mask=tuple(bool(value) for value in voiced_mask),
        frame_rms_dbfs=tuple(float(value) for value in frame_rms_dbfs),
        periodicity=tuple(float(value) for value in periodicity),
        frame_to_frame_differences_hz=first_differences,
        second_order_changes_hz=second_changes,
    )
    return F0AnalysisResult(
        status=status,
        algorithm=active_settings.algorithm,
        sample_rate_hz=rate,
        frame_length_samples=active_settings.frame_length_samples,
        hop_length_samples=active_settings.hop_length_samples,
        variation_is_reliable=variation_is_reliable,
        metrics=metrics,
        track=track,
        rule_evaluations=_build_rule_evaluations(
            metrics,
            variation_is_reliable=variation_is_reliable,
            settings=active_settings,
        ),
        warnings=warnings,
    )


def calculate_f0_differences(
    f0_hz: ArrayLike,
    voiced_mask: Sequence[bool] | NDArray[np.bool_],
) -> tuple[tuple[float | None, ...], tuple[float | None, ...]]:
    """Calculate signed first/second F0 changes for contiguous voiced frames.

    Returned tuples are frame-aligned. The first element of a first-difference
    track and the first two elements of a second-difference track are ``None``;
    gaps are also ``None`` so unvoiced regions are never bridged.
    """

    values = np.asarray(f0_hz, dtype=np.float64)
    mask = np.asarray(voiced_mask)
    if values.ndim != 1 or mask.ndim != 1:
        raise ValueError("f0_hz and voiced_mask must be one-dimensional")
    if values.size != mask.size:
        raise ValueError("f0_hz and voiced_mask must have the same length")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError("voiced_mask must contain boolean values")
    if np.any(mask & ~np.isfinite(values)):
        raise ValueError("voiced F0 values must be finite")

    first: list[float | None] = [None] * values.size
    second: list[float | None] = [None] * values.size
    for index in range(1, values.size):
        if mask[index] and mask[index - 1]:
            first[index] = float(values[index] - values[index - 1])
    for index in range(2, values.size):
        if mask[index] and mask[index - 1] and mask[index - 2]:
            second[index] = float(
                values[index] - 2.0 * values[index - 1] + values[index - 2]
            )
    return tuple(first), tuple(second)


def _calculate_metrics(
    candidates: NDArray[np.floating[Any]],
    voiced_mask: NDArray[np.bool_],
    first_differences: tuple[float | None, ...],
    second_changes: tuple[float | None, ...],
    *,
    ddof: int,
) -> F0Metrics:
    voiced_values = candidates[voiced_mask].astype(np.float64, copy=False)
    total_count = int(candidates.size)
    voiced_count = int(voiced_values.size)
    voiced_ratio = voiced_count / total_count if total_count else 0.0
    valid_first = [value for value in first_differences if value is not None]
    valid_second = [value for value in second_changes if value is not None]

    if voiced_count == 0:
        mean_f0 = minimum_f0 = maximum_f0 = None
    else:
        mean_f0 = float(np.mean(voiced_values))
        minimum_f0 = float(np.min(voiced_values))
        maximum_f0 = float(np.max(voiced_values))
    if voiced_count < 2:
        std_f0 = f0_range = None
    else:
        std_f0 = float(np.std(voiced_values, ddof=ddof))
        f0_range = float(np.ptp(voiced_values))

    return F0Metrics(
        total_frame_count=total_count,
        voiced_frame_count=voiced_count,
        voiced_frame_ratio=float(voiced_ratio),
        mean_f0_hz=mean_f0,
        std_f0_hz=std_f0,
        minimum_f0_hz=minimum_f0,
        maximum_f0_hz=maximum_f0,
        f0_range_hz=f0_range,
        mean_abs_frame_difference_hz=(
            float(np.mean(np.abs(valid_first))) if valid_first else None
        ),
        mean_abs_second_order_change_hz=(
            float(np.mean(np.abs(valid_second))) if valid_second else None
        ),
        frame_difference_count=len(valid_first),
        second_order_change_count=len(valid_second),
    )


def _periodicity_at_candidate(
    frame: NDArray[np.floating[Any]],
    candidate_f0_hz: float,
    sample_rate: int,
) -> float:
    if not np.isfinite(candidate_f0_hz) or candidate_f0_hz <= 0:
        return 0.0
    lag = int(round(sample_rate / float(candidate_f0_hz)))
    if lag <= 0 or lag >= frame.size:
        return 0.0
    centered = frame.astype(np.float64, copy=False) - float(np.mean(frame))
    left = centered[:-lag]
    right = centered[lag:]
    denominator = float(
        np.sqrt(np.dot(left, left) * np.dot(right, right))
    )
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0
    correlation = float(np.dot(left, right) / denominator)
    return float(np.clip(correlation, 0.0, 1.0))


def _remove_short_voiced_runs(
    mask: NDArray[np.bool_], minimum_run_frames: int
) -> NDArray[np.bool_]:
    filtered = np.asarray(mask, dtype=np.bool_).copy()
    run_start: int | None = None
    for index in range(filtered.size + 1):
        is_voiced = index < filtered.size and bool(filtered[index])
        if is_voiced and run_start is None:
            run_start = index
        elif not is_voiced and run_start is not None:
            if index - run_start < minimum_run_frames:
                filtered[run_start:index] = False
            run_start = None
    return filtered


def _build_rule_evaluations(
    metrics: F0Metrics,
    *,
    variation_is_reliable: bool,
    settings: F0Settings,
    voiced_ratio_is_evaluable: bool = True,
) -> tuple[F0RuleEvaluation, ...]:
    low_voiced_observed = metrics.voiced_frame_ratio
    evaluations = [
        F0RuleEvaluation(
            code="LOW_VOICED_RATIO",
            observed_value=(
                low_voiced_observed if voiced_ratio_is_evaluable else None
            ),
            threshold=settings.low_voiced_ratio_below,
            comparison="below",
            triggered=(
                low_voiced_observed < settings.low_voiced_ratio_below
                if voiced_ratio_is_evaluable
                else None
            ),
            explanation="Voiced content is below the configured reliability ratio.",
        )
    ]
    variation_rules = (
        (
            "LOW_F0_STD",
            metrics.std_f0_hz,
            settings.low_std_hz,
            "Very low F0 standard deviation.",
        ),
        (
            "LOW_F0_RANGE",
            metrics.f0_range_hz,
            settings.low_range_hz,
            "Very small literal maximum-minus-minimum F0 range.",
        ),
        (
            "LOW_F0_FRAME_DIFFERENCE",
            metrics.mean_abs_frame_difference_hz,
            settings.low_mean_abs_delta_hz_per_frame,
            "Very small mean absolute frame-to-frame F0 difference.",
        ),
        (
            "LOW_F0_SECOND_ORDER_CHANGE",
            metrics.mean_abs_second_order_change_hz,
            settings.low_mean_abs_second_delta_hz_per_frame2,
            "Very small mean absolute second-order F0 change.",
        ),
    )
    for code, observed, threshold, explanation in variation_rules:
        evaluated_value = observed if variation_is_reliable else None
        evaluations.append(
            F0RuleEvaluation(
                code=code,
                observed_value=evaluated_value,
                threshold=threshold,
                comparison="below",
                triggered=(
                    evaluated_value < threshold
                    if evaluated_value is not None
                    else None
                ),
                explanation=explanation,
            )
        )
    return tuple(evaluations)


def _empty_result(
    status: F0AnalysisStatus,
    sample_rate: int,
    settings: F0Settings,
    warning: str,
) -> F0AnalysisResult:
    metrics = F0Metrics(
        total_frame_count=0,
        voiced_frame_count=0,
        voiced_frame_ratio=0.0,
        mean_f0_hz=None,
        std_f0_hz=None,
        minimum_f0_hz=None,
        maximum_f0_hz=None,
        f0_range_hz=None,
        mean_abs_frame_difference_hz=None,
        mean_abs_second_order_change_hz=None,
        frame_difference_count=0,
        second_order_change_count=0,
    )
    track = F0Track(
        frame_start_times_seconds=(),
        candidate_f0_hz=(),
        voiced_f0_hz=(),
        voiced_mask=(),
        frame_rms_dbfs=(),
        periodicity=(),
        frame_to_frame_differences_hz=(),
        second_order_changes_hz=(),
    )
    return F0AnalysisResult(
        status=status,
        algorithm=settings.algorithm,
        sample_rate_hz=sample_rate,
        frame_length_samples=settings.frame_length_samples,
        hop_length_samples=settings.hop_length_samples,
        variation_is_reliable=False,
        metrics=metrics,
        track=track,
        rule_evaluations=_build_rule_evaluations(
            metrics,
            variation_is_reliable=False,
            settings=settings,
            voiced_ratio_is_evaluable=False,
        ),
        warnings=(warning,),
    )


def _amplitude_to_dbfs(values: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    safe_values = np.maximum(values.astype(np.float64, copy=False), np.finfo(float).tiny)
    return 20.0 * np.log10(safe_values)


def _validate_positive_integer(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _validate_finite_real(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_positive_real(value: float, *, name: str) -> None:
    _validate_finite_real(value, name=name)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _validate_non_negative_real(value: float, *, name: str) -> None:
    _validate_finite_real(value, name=name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_unit_interval(
    value: float,
    *,
    name: str,
    exclude_zero: bool = False,
) -> None:
    _validate_finite_real(value, name=name)
    lower_invalid = value <= 0 if exclude_zero else value < 0
    if lower_invalid or value > 1:
        interval = "(0, 1]" if exclude_zero else "[0, 1]"
        raise ValueError(f"{name} must be in {interval}")
