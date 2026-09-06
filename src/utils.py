"""Shared validation, configuration, and numerical helpers for SignalGuard."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from math import ceil
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
_NUMERICAL_FLOOR = np.finfo(np.float64).tiny


class ConfigError(ValueError):
    """Raised when a SignalGuard configuration file is invalid."""


def load_config(
    path: str | Path | None = None, *, validate: bool = True
) -> dict[str, Any]:
    """Load a YAML configuration whose top level must be a mapping.

    The default location is resolved relative to the installed source tree, not
    the process working directory. This makes SDK behavior independent of where
    the caller launched Python. Full project invariants are validated by
    default; pass ``validate=False`` only when intentionally parsing a partial
    configuration fragment.
    """

    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    try:
        raw_config = config_path.read_text(encoding="utf-8")
    except OSError:
        raise

    try:
        loaded = yaml.safe_load(raw_config)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in configuration: {config_path}") from exc

    if not isinstance(loaded, Mapping):
        raise ConfigError(
            f"Configuration must contain a top-level mapping: {config_path}"
        )
    config = dict(loaded)
    if validate:
        validate_project_config(config)
    return config


def get_config_section(
    config: Mapping[str, Any], section_name: str
) -> Mapping[str, Any]:
    """Return a required mapping section from a loaded configuration."""

    try:
        section = config[section_name]
    except KeyError as exc:
        raise ConfigError(f"Missing configuration section: {section_name}") from exc
    if not isinstance(section, Mapping):
        raise ConfigError(f"Configuration section '{section_name}' must be a mapping")
    return section


def validate_project_config(config: Mapping[str, Any]) -> None:
    """Validate cross-section invariants in the full SignalGuard config.

    Detector thresholds remain heuristics, but malformed ordering, impossible
    frequency values, invalid kernels, and scoring weights that do not total 100
    are configuration errors rather than calibration choices.
    """

    if not isinstance(config, Mapping):
        raise ConfigError("Project configuration must be a mapping")

    preprocessing = get_config_section(config, "preprocessing")
    target_rate = _required_integer(
        preprocessing,
        "target_sample_rate",
        path="preprocessing.target_sample_rate",
    )
    maximum_rate = _required_integer(
        preprocessing,
        "maximum_input_sample_rate",
        path="preprocessing.maximum_input_sample_rate",
    )
    if target_rate <= 0 or maximum_rate < target_rate:
        raise ConfigError(
            "preprocessing sample rates must be positive and the maximum must "
            "not be below the target"
        )
    maximum_factor = _required_real(
        preprocessing,
        "maximum_resample_factor",
        path="preprocessing.maximum_resample_factor",
    )
    if maximum_factor < 1.0:
        raise ConfigError("preprocessing.maximum_resample_factor must be at least 1")
    maximum_polyphase_factor = _required_integer(
        preprocessing,
        "maximum_polyphase_factor",
        path="preprocessing.maximum_polyphase_factor",
    )
    if maximum_polyphase_factor <= 0:
        raise ConfigError("preprocessing.maximum_polyphase_factor must be positive")
    target_peak = _required_real(
        preprocessing,
        "target_peak",
        path="preprocessing.target_peak",
    )
    if target_peak <= 0 or target_peak > 1:
        raise ConfigError("preprocessing.target_peak must be in the interval (0, 1]")
    for key in ("segment_duration_seconds",):
        if _required_real(preprocessing, key, path=f"preprocessing.{key}") <= 0:
            raise ConfigError(f"preprocessing.{key} must be positive")
    if (
        _required_real(
            preprocessing,
            "max_normalization_gain_db",
            path="preprocessing.max_normalization_gain_db",
        )
        < 0
    ):
        raise ConfigError("preprocessing.max_normalization_gain_db cannot be negative")
    if preprocessing.get("normalization_mode") not in ("peak_limit", "peak"):
        raise ConfigError(
            "preprocessing.normalization_mode must be 'peak_limit' or 'peak'"
        )
    for boolean_name in (
        "remove_dc_offset",
        "include_partial_segment",
        "pad_final_segment",
    ):
        if not isinstance(preprocessing.get(boolean_name), bool):
            raise ConfigError(f"preprocessing.{boolean_name} must be bool")
    if (
        preprocessing["pad_final_segment"]
        and not preprocessing["include_partial_segment"]
    ):
        raise ConfigError(
            "preprocessing.pad_final_segment requires include_partial_segment"
        )

    forensics = get_config_section(config, "forensics")
    boundaries = get_config_section(forensics, "decision_thresholds")
    ordered_boundaries = [
        _required_real(boundaries, key, path=f"forensics.decision_thresholds.{key}")
        for key in (
            "likely_natural_min",
            "weak_evidence_min",
            "review_min",
            "likely_synthetic_min",
        )
    ]
    if ordered_boundaries[0] != 0.0 or any(
        left >= right
        for left, right in zip(ordered_boundaries, ordered_boundaries[1:])
    ):
        raise ConfigError(
            "forensic decision lower bounds must start at 0 and be strictly increasing"
        )
    if ordered_boundaries[-1] > 100.0:
        raise ConfigError("forensic decision bounds must not exceed 100")

    weight_groups: dict[str, Mapping[str, Any]] = {
        "forensics.scoring_weights": get_config_section(
            forensics, "scoring_weights"
        )
    }
    for detector_name in (
        "f0",
        "harmonic",
        "spectral",
        "phase",
        "temporal",
        "lpc",
        "bispectrum",
        "modulation",
        "breath",
        "decay",
    ):
        if detector_name in forensics:
            detector = get_config_section(forensics, detector_name)
            weight_groups[f"forensics.{detector_name}.scoring_weights"] = (
                get_config_section(detector, "scoring_weights")
            )
    for path, weights in weight_groups.items():
        values = [
            _coerce_real(value, path=f"{path}.{name}")
            for name, value in weights.items()
        ]
        if not values or any(value < 0 for value in values):
            raise ConfigError(f"{path} must contain non-negative weights")
        if not np.isclose(sum(values), 100.0, rtol=0.0, atol=1e-6):
            raise ConfigError(f"{path} weights must total 100")

    for detector_name, frame_key in (
        ("harmonic", "frame_length"),
        ("spectral", "n_fft"),
        ("phase", "n_fft"),
        ("bispectrum", "n_fft"),
    ):
        if detector_name in forensics:
            detector = get_config_section(forensics, detector_name)
            frame_size = _required_integer(
                detector,
                frame_key,
                path=f"forensics.{detector_name}.{frame_key}",
            )
            detector_hop = _required_integer(
                detector,
                "hop_length",
                path=f"forensics.{detector_name}.hop_length",
            )
            if frame_size <= 0 or detector_hop <= 0 or detector_hop > frame_size:
                raise ConfigError(
                    f"forensics.{detector_name} frame/hop lengths must be positive "
                    "and hop must not exceed the frame length"
                )

    aggregation = get_config_section(forensics, "segment_aggregation")
    if aggregation.get("method") != "mean_with_max_guard":
        raise ConfigError(
            "forensics.segment_aggregation.method must be 'mean_with_max_guard'"
        )
    guard_score = _required_real(
        aggregation,
        "max_segment_guard_score",
        path="forensics.segment_aggregation.max_segment_guard_score",
    )
    if not 0 <= guard_score <= 100:
        raise ConfigError("segment max guard score must be between 0 and 100")

    harmonic_section = get_config_section(forensics, "harmonic")
    harmonic_count = _required_integer(
        harmonic_section,
        "harmonic_count",
        path="forensics.harmonic.harmonic_count",
    )
    harmonic_tolerance = _required_real(
        harmonic_section,
        "bin_tolerance_hz",
        path="forensics.harmonic.bin_tolerance_hz",
    )
    hnr_low = _required_real(
        harmonic_section,
        "hnr_low_db",
        path="forensics.harmonic.hnr_low_db",
    )
    hnr_high = _required_real(
        harmonic_section,
        "hnr_high_db",
        path="forensics.harmonic.hnr_high_db",
    )
    if harmonic_count <= 0 or harmonic_tolerance <= 0 or hnr_low >= hnr_high:
        raise ConfigError(
            "harmonic count/tolerance must be positive and hnr_low_db must be below hnr_high_db"
        )

    spectral_section = get_config_section(forensics, "spectral")
    rolloff_percent = _required_real(
        spectral_section,
        "rolloff_percent",
        path="forensics.spectral.rolloff_percent",
    )
    if not 0 < rolloff_percent < 1:
        raise ConfigError("forensics.spectral.rolloff_percent must be in (0, 1)")
    for key in (
        "low_centroid_cv",
        "low_bandwidth_cv",
        "low_flux_cv",
        "low_rolloff_cv",
    ):
        if _required_real(spectral_section, key, path=f"forensics.spectral.{key}") < 0:
            raise ConfigError(f"forensics.spectral.{key} must be non-negative")

    phase_section = get_config_section(forensics, "phase")
    phase_jump = _required_real(
        phase_section,
        "phase_jump_radians",
        path="forensics.phase.phase_jump_radians",
    )
    phase_variation = _required_real(
        phase_section,
        "low_phase_variation",
        path="forensics.phase.low_phase_variation",
    )
    group_delay_zscore = _required_real(
        phase_section,
        "group_delay_outlier_zscore",
        path="forensics.phase.group_delay_outlier_zscore",
    )
    if not 0 < phase_jump <= np.pi:
        raise ConfigError("forensics.phase.phase_jump_radians must be in (0, pi]")
    if phase_variation < 0 or group_delay_zscore <= 0:
        raise ConfigError(
            "phase variation must be non-negative and group-delay z-score positive"
        )

    temporal_section = get_config_section(forensics, "temporal")
    if _required_integer(
        temporal_section,
        "min_segments",
        path="forensics.temporal.min_segments",
    ) <= 0:
        raise ConfigError("forensics.temporal.min_segments must be positive")
    for key in ("low_energy_cv", "low_centroid_cv", "low_zcr_cv", "low_f0_cv"):
        if _required_real(temporal_section, key, path=f"forensics.temporal.{key}") < 0:
            raise ConfigError(f"forensics.temporal.{key} must be non-negative")

    nyquist_hz = target_rate / 2.0
    frequency_paths = (
        ("forensics", "f0", "fmax_hz"),
        ("corruption_detection", "hiss", "lower_frequency_hz"),
        ("restoration", "hiss_filter", "cutoff_hz"),
    )
    for path_parts in frequency_paths:
        parent = _nested_section(config, *path_parts[:-1])
        frequency = _required_real(
            parent,
            path_parts[-1],
            path=".".join(path_parts),
        )
        if frequency <= 0 or frequency >= nyquist_hz:
            raise ConfigError(
                f"{'.'.join(path_parts)} must be positive and below Nyquist "
                f"({nyquist_hz:g} Hz)"
            )

    f0_section = _nested_section(config, "forensics", "f0")
    fmin_hz = _required_real(
        f0_section,
        "fmin_hz",
        path="forensics.f0.fmin_hz",
    )
    fmax_hz = _required_real(
        f0_section,
        "fmax_hz",
        path="forensics.f0.fmax_hz",
    )
    if fmin_hz <= 0 or fmin_hz >= fmax_hz:
        raise ConfigError("forensics F0 limits must satisfy 0 < fmin_hz < fmax_hz")
    if f0_section.get("algorithm") != "yin":
        raise ConfigError("forensics.f0.algorithm must be 'yin'")
    frame_length = _required_integer(
        f0_section,
        "frame_length_samples",
        path="forensics.f0.frame_length_samples",
    )
    hop_length = _required_integer(
        f0_section,
        "hop_length_samples",
        path="forensics.f0.hop_length_samples",
    )
    if frame_length <= 0 or hop_length <= 0 or hop_length > frame_length:
        raise ConfigError(
            "F0 frame/hop lengths must be positive and hop must not exceed frame"
        )
    if frame_length < ceil(2.0 * target_rate / fmin_hz):
        raise ConfigError("F0 frame length must span at least two periods at fmin_hz")
    trough_threshold = _required_real(
        f0_section,
        "trough_threshold",
        path="forensics.f0.trough_threshold",
    )
    if trough_threshold <= 0 or trough_threshold > 1:
        raise ConfigError("forensics.f0.trough_threshold must be in (0, 1]")
    if f0_section.get("center") is not False:
        raise ConfigError("forensics.f0.center must be false")
    if f0_section.get("postprocessing") != "none":
        raise ConfigError("forensics.f0.postprocessing must be 'none'")
    if f0_section.get("pad_mode") != "constant":
        raise ConfigError("forensics.f0.pad_mode must be 'constant'")
    if f0_section.get("range_method") != "max_minus_min":
        raise ConfigError("forensics.f0.range_method must be 'max_minus_min'")
    if (
        _required_integer(
            f0_section,
            "std_ddof",
            path="forensics.f0.std_ddof",
        )
        != 0
    ):
        raise ConfigError("forensics.f0.std_ddof must be 0")
    if f0_section.get("delta_policy") != "contiguous_voiced_only":
        raise ConfigError(
            "forensics.f0.delta_policy must be 'contiguous_voiced_only'"
        )
    rms_floor = _required_real(
        f0_section,
        "rms_floor_dbfs",
        path="forensics.f0.rms_floor_dbfs",
    )
    relative_rms_range = _required_real(
        f0_section,
        "relative_rms_range_db",
        path="forensics.f0.relative_rms_range_db",
    )
    rms_percentile = _required_real(
        f0_section,
        "rms_reference_percentile",
        path="forensics.f0.rms_reference_percentile",
    )
    if rms_floor > 0 or relative_rms_range <= 0:
        raise ConfigError("F0 RMS thresholds must describe a valid dBFS range")
    if rms_percentile <= 0 or rms_percentile > 100:
        raise ConfigError("F0 RMS reference percentile must be in (0, 100]")
    for key in (
        "minimum_periodicity",
        "minimum_voiced_ratio_for_variation",
        "low_voiced_ratio_below",
    ):
        value = _required_real(f0_section, key, path=f"forensics.f0.{key}")
        if value < 0 or value > 1:
            raise ConfigError(f"forensics.f0.{key} must be between 0 and 1")
    for key in (
        "minimum_voiced_run_frames",
        "minimum_voiced_frames_for_variation",
    ):
        value = _required_integer(f0_section, key, path=f"forensics.f0.{key}")
        if value <= 0:
            raise ConfigError(f"forensics.f0.{key} must be positive")
    if (
        _required_integer(
            f0_section,
            "minimum_voiced_frames_for_variation",
            path="forensics.f0.minimum_voiced_frames_for_variation",
        )
        < 3
    ):
        raise ConfigError("F0 variation needs at least three voiced frames")
    for key in (
        "low_std_hz",
        "low_range_hz",
        "low_mean_abs_delta_hz_per_frame",
        "low_mean_abs_second_delta_hz_per_frame2",
    ):
        value = _required_real(f0_section, key, path=f"forensics.f0.{key}")
        if value < 0:
            raise ConfigError(f"forensics.f0.{key} must be non-negative")

    hum_section = _nested_section(config, "corruption_detection", "hum")
    try:
        mains_frequencies = hum_section["supported_mains_frequencies_hz"]
    except KeyError as exc:
        raise ConfigError(
            "Missing configuration value: "
            "corruption_detection.hum.supported_mains_frequencies_hz"
        ) from exc
    if not isinstance(mains_frequencies, list) or not mains_frequencies:
        raise ConfigError("supported mains frequencies must be a non-empty list")
    harmonic_count = _required_integer(
        hum_section,
        "harmonic_count",
        path="corruption_detection.hum.harmonic_count",
    )
    parsed_mains = [
        _coerce_real(
            frequency,
            path=f"corruption_detection.hum.supported_mains_frequencies_hz[{index}]",
        )
        for index, frequency in enumerate(mains_frequencies)
    ]
    if harmonic_count <= 0 or any(
        frequency <= 0 or frequency * harmonic_count >= nyquist_hz
        for frequency in parsed_mains
    ):
        raise ConfigError(
            "hum frequencies and harmonics must be positive and below Nyquist"
        )
    mains_frequency = _required_real(
        hum_section,
        "mains_frequency_hz",
        path="corruption_detection.hum.mains_frequency_hz",
    )
    if not any(np.isclose(mains_frequency, item) for item in parsed_mains):
        raise ConfigError(
            "corruption_detection.hum.mains_frequency_hz must be one of the supported frequencies"
        )

    broadband_section = _nested_section(
        config,
        "corruption_detection",
        "broadband_noise",
    )
    broadband_frame = _required_integer(
        broadband_section,
        "frame_length",
        path="corruption_detection.broadband_noise.frame_length",
    )
    broadband_hop = _required_integer(
        broadband_section,
        "hop_length",
        path="corruption_detection.broadband_noise.hop_length",
    )
    broadband_quantile = _required_real(
        broadband_section,
        "low_energy_quantile",
        path="corruption_detection.broadband_noise.low_energy_quantile",
    )
    if (
        broadband_frame <= 0
        or broadband_hop <= 0
        or broadband_hop > broadband_frame
        or not 0 < broadband_quantile <= 1
    ):
        raise ConfigError(
            "broadband-noise frame/hop must be valid and quantile must be in (0, 1]"
        )

    clipping_level = _required_path_real(
        config,
        ("corruption_detection", "clipping", "sample_level"),
    )
    if not 0 < clipping_level <= 1:
        raise ConfigError(
            "corruption_detection.clipping.sample_level must be in (0, 1]"
        )
    clipping_tolerance = _required_path_real(
        config,
        ("corruption_detection", "clipping", "flat_top_tolerance"),
    )
    if not 0 <= clipping_tolerance <= 1:
        raise ConfigError(
            "corruption_detection.clipping.flat_top_tolerance must be between 0 and 1"
        )
    for key in ("minimum_absolute_amplitude", "minimum_residual_amplitude"):
        impulse_amplitude = _required_path_real(
            config,
            ("corruption_detection", "impulse", key),
        )
        if not 0 <= impulse_amplitude <= 1:
            raise ConfigError(
                f"corruption_detection.impulse.{key} must be between 0 and 1"
            )

    noise_reduction = _nested_section(config, "restoration", "noise_reduction")
    if noise_reduction.get("method") != "spectral_subtraction":
        raise ConfigError(
            "restoration.noise_reduction.method must be 'spectral_subtraction'"
        )
    noise_fft = _required_integer(
        noise_reduction,
        "n_fft",
        path="restoration.noise_reduction.n_fft",
    )
    noise_hop = _required_integer(
        noise_reduction,
        "hop_length",
        path="restoration.noise_reduction.hop_length",
    )
    if noise_fft <= 0 or noise_hop <= 0 or noise_hop > noise_fft:
        raise ConfigError(
            "restoration noise-reduction FFT/hop lengths are invalid"
        )
    for key in ("subtraction_factor", "spectral_floor_ratio"):
        if _required_real(
            noise_reduction,
            key,
            path=f"restoration.noise_reduction.{key}",
        ) < 0:
            raise ConfigError(
                f"restoration.noise_reduction.{key} must be non-negative"
            )
    if float(noise_reduction["spectral_floor_ratio"]) > 1:
        raise ConfigError(
            "restoration.noise_reduction.spectral_floor_ratio must not exceed 1"
        )

    positive_real_paths = (
        ("restoration", "notch_filter", "quality_factor"),
        ("corruption_detection", "dropout", "minimum_duration_ms"),
        ("corruption_detection", "dropout", "maximum_restorable_duration_ms"),
        ("quality_validation", "minimum_snr_improvement_db"),
    )
    for path_parts in positive_real_paths:
        if _required_path_real(config, path_parts) <= 0:
            raise ConfigError(f"{'.'.join(path_parts)} must be positive")

    minimum_dropout_ms = _required_path_real(
        config,
        ("corruption_detection", "dropout", "minimum_duration_ms"),
    )
    maximum_dropout_ms = _required_path_real(
        config,
        ("corruption_detection", "dropout", "maximum_restorable_duration_ms"),
    )
    if maximum_dropout_ms < minimum_dropout_ms:
        raise ConfigError(
            "maximum restorable dropout duration must not be below the minimum "
            "detected duration"
        )

    positive_integer_paths = (
        ("restoration", "rumble_filter", "order"),
        ("restoration", "hiss_filter", "order"),
        ("corruption_detection", "clipping", "flat_top_min_samples"),
    )
    for path_parts in positive_integer_paths:
        parent = _nested_section(config, *path_parts[:-1])
        value = _required_integer(
            parent,
            path_parts[-1],
            path=".".join(path_parts),
        )
        if value <= 0:
            raise ConfigError(f"{'.'.join(path_parts)} must be positive")

    ascending_severity_groups = (
        (
            "clipping ratios",
            (
                ("corruption_detection", "clipping", "mild_ratio"),
                ("corruption_detection", "clipping", "moderate_ratio"),
                ("corruption_detection", "clipping", "severe_ratio"),
            ),
        ),
        (
            "hum prominence",
            (
                (
                    "corruption_detection",
                    "hum",
                    "minimum_peak_to_local_ratio_db",
                ),
                (
                    "corruption_detection",
                    "hum",
                    "severity_boundaries_prominence_db",
                    "moderate_above",
                ),
                (
                    "corruption_detection",
                    "hum",
                    "severity_boundaries_prominence_db",
                    "high_above",
                ),
            ),
        ),
        (
            "impulse event rates",
            (
                (
                    "corruption_detection",
                    "impulse",
                    "detection_events_per_second",
                ),
                (
                    "corruption_detection",
                    "impulse",
                    "severity_boundaries_events_per_second",
                    "moderate_above",
                ),
                (
                    "corruption_detection",
                    "impulse",
                    "severity_boundaries_events_per_second",
                    "high_above",
                ),
            ),
        ),
        (
            "dropout ratios",
            (
                ("corruption_detection", "dropout", "detection_total_ratio"),
                (
                    "corruption_detection",
                    "dropout",
                    "severity_boundaries_total_ratio",
                    "moderate_above",
                ),
                (
                    "corruption_detection",
                    "dropout",
                    "severity_boundaries_total_ratio",
                    "high_above",
                ),
            ),
        ),
        (
            "rumble energy ratios",
            (
                ("corruption_detection", "rumble", "excessive_energy_ratio"),
                (
                    "corruption_detection",
                    "rumble",
                    "severity_boundaries_energy_ratio",
                    "moderate_above",
                ),
                (
                    "corruption_detection",
                    "rumble",
                    "severity_boundaries_energy_ratio",
                    "high_above",
                ),
            ),
        ),
        (
            "hiss energy ratios",
            (
                ("corruption_detection", "hiss", "excessive_energy_ratio"),
                (
                    "corruption_detection",
                    "hiss",
                    "severity_boundaries_energy_ratio",
                    "moderate_above",
                ),
                (
                    "corruption_detection",
                    "hiss",
                    "severity_boundaries_energy_ratio",
                    "high_above",
                ),
            ),
        ),
    )
    for label, paths in ascending_severity_groups:
        values = [_required_path_real(config, path) for path in paths]
        if any(left >= right for left, right in zip(values, values[1:])):
            raise ConfigError(f"{label} must be strictly increasing")

    noise_snr_paths = (
        ("corruption_detection", "broadband_noise", "detection_snr_db"),
        (
            "corruption_detection",
            "broadband_noise",
            "severity_boundaries_snr_db",
            "moderate_below",
        ),
        (
            "corruption_detection",
            "broadband_noise",
            "severity_boundaries_snr_db",
            "high_below",
        ),
    )
    noise_snr_values = [
        _required_path_real(config, path) for path in noise_snr_paths
    ]
    if any(
        left <= right
        for left, right in zip(noise_snr_values, noise_snr_values[1:])
    ):
        raise ConfigError("broadband-noise SNR boundaries must be strictly decreasing")

    bounded_ratio_paths = (
        ("corruption_detection", "clipping", "mild_ratio"),
        ("corruption_detection", "clipping", "moderate_ratio"),
        ("corruption_detection", "clipping", "severe_ratio"),
        ("corruption_detection", "dropout", "detection_total_ratio"),
        (
            "corruption_detection",
            "dropout",
            "severity_boundaries_total_ratio",
            "moderate_above",
        ),
        (
            "corruption_detection",
            "dropout",
            "severity_boundaries_total_ratio",
            "high_above",
        ),
        ("corruption_detection", "rumble", "excessive_energy_ratio"),
        (
            "corruption_detection",
            "rumble",
            "severity_boundaries_energy_ratio",
            "moderate_above",
        ),
        (
            "corruption_detection",
            "rumble",
            "severity_boundaries_energy_ratio",
            "high_above",
        ),
        ("corruption_detection", "hiss", "excessive_energy_ratio"),
        (
            "corruption_detection",
            "hiss",
            "severity_boundaries_energy_ratio",
            "moderate_above",
        ),
        (
            "corruption_detection",
            "hiss",
            "severity_boundaries_energy_ratio",
            "high_above",
        ),
        ("quality_validation", "maximum_clipping_ratio"),
        ("forensics", "harmonic", "low_harmonic_energy_ratio"),
        ("forensics", "harmonic", "high_out_of_harmonic_ratio"),
        ("forensics", "spectral", "high_flatness"),
        ("forensics", "spectral", "high_normalized_entropy"),
        ("forensics", "spectral", "high_zcr"),
        ("forensics", "phase", "high_phase_jump_ratio"),
        ("forensics", "phase", "minimum_bin_magnitude_ratio"),
        ("forensics", "phase", "high_group_delay_outlier_ratio"),
        ("restoration", "noise_reduction", "noise_reference_quantile"),
    )
    for path_parts in bounded_ratio_paths:
        ratio = _required_path_real(config, path_parts)
        if ratio < 0 or ratio > 1:
            raise ConfigError(f"{'.'.join(path_parts)} must be between 0 and 1")

    noise_reference_quantile = _required_path_real(
        config,
        ("restoration", "noise_reduction", "noise_reference_quantile"),
    )
    if noise_reference_quantile == 0:
        raise ConfigError(
            "restoration.noise_reduction.noise_reference_quantile must be greater than zero"
        )
    for path_parts in (
        ("forensics", "phase", "minimum_bin_magnitude_ratio"),
        ("forensics", "phase", "high_group_delay_outlier_ratio"),
    ):
        if _required_path_real(config, path_parts) == 0:
            raise ConfigError(f"{'.'.join(path_parts)} must be greater than zero")

    success_section = _nested_section(
        config,
        "quality_validation",
        "success_by_corruption",
    )
    required_success_metrics: dict[str, tuple[str, ...]] = {
        "broadband_noise": (
            "minimum_snr_improvement_db",
            "minimum_post_snr_db",
        ),
        "hum": ("minimum_prominence_reduction_db",),
        "impulse": ("minimum_event_reduction_ratio",),
        "dropout": ("minimum_duration_reduction_ratio",),
        "rumble": ("minimum_band_energy_reduction_ratio",),
        "hiss": ("minimum_band_energy_reduction_ratio",),
        "clipping": ("minimum_clipping_reduction_ratio",),
    }
    ratio_success_metrics = {
        ("impulse", "minimum_event_reduction_ratio"),
        ("dropout", "minimum_duration_reduction_ratio"),
        ("rumble", "minimum_band_energy_reduction_ratio"),
        ("hiss", "minimum_band_energy_reduction_ratio"),
        ("clipping", "minimum_clipping_reduction_ratio"),
    }
    for corruption_name, metric_names in required_success_metrics.items():
        corruption_metrics = get_config_section(success_section, corruption_name)
        for metric_name in metric_names:
            value = _required_real(
                corruption_metrics,
                metric_name,
                path=(
                    "quality_validation.success_by_corruption."
                    f"{corruption_name}.{metric_name}"
                ),
            )
            if value <= 0:
                raise ConfigError(
                    "restoration success metrics must be greater than zero"
                )
            if (corruption_name, metric_name) in ratio_success_metrics and value > 1:
                raise ConfigError(
                    "restoration success reduction ratios must not exceed 1"
                )

    odd_kernel_paths = (
        ("corruption_detection", "impulse", "local_window_samples"),
        ("restoration", "impulse_repair", "median_kernel_size"),
    )
    for path_parts in odd_kernel_paths:
        parent = _nested_section(config, *path_parts[:-1])
        kernel_size = _required_integer(
            parent,
            path_parts[-1],
            path=".".join(path_parts),
        )
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ConfigError(f"{'.'.join(path_parts)} must be a positive odd integer")


def validate_sample_rate(sample_rate: int, *, name: str = "sample_rate") -> int:
    """Validate and return a positive integer sample rate."""

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral):
        raise TypeError(f"{name} must be a positive integer")
    validated = int(sample_rate)
    if validated <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return validated


def validate_audio_array(
    samples: ArrayLike,
    *,
    allow_empty: bool = False,
    allowed_dimensions: tuple[int, ...] = (1, 2),
    name: str = "samples",
) -> np.ndarray:
    """Validate an audio-like array without changing its dtype or shape."""

    array = np.asarray(samples)
    if array.ndim not in allowed_dimensions:
        dimensions = ", ".join(str(value) for value in allowed_dimensions)
        raise ValueError(f"{name} must have one of these dimensions: {dimensions}")
    if array.size == 0 and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must contain real numeric audio samples, not bool")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must contain real numeric audio samples")
    if array.size and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def signal_power(samples: ArrayLike) -> float:
    """Return mean-square signal power using float64 accumulation."""

    array = validate_audio_array(samples, allowed_dimensions=(1,))
    values = array.astype(np.float64, copy=False)
    return float(np.mean(np.square(values)))


def rms(samples: ArrayLike) -> float:
    """Return the root-mean-square amplitude of a non-empty mono signal."""

    return float(np.sqrt(signal_power(samples)))


def power_to_db(
    power: float | NDArray[np.floating[Any]],
    *,
    reference: float = 1.0,
    floor_db: float = -120.0,
) -> float | NDArray[np.float64]:
    """Convert non-negative power to decibels with a finite lower floor."""

    if isinstance(reference, bool) or not isinstance(reference, Real):
        raise TypeError("reference must be a positive real number")
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("reference must be finite and greater than zero")
    if isinstance(floor_db, bool) or not isinstance(floor_db, Real):
        raise TypeError("floor_db must be a real number")
    if not np.isfinite(floor_db) or floor_db > 0:
        raise ValueError("floor_db must be finite and no greater than zero")

    values = np.asarray(power, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("power must contain finite, non-negative values")

    safe_power = np.maximum(values, _NUMERICAL_FLOOR)
    decibels = 10.0 * (
        np.log10(safe_power) - np.log10(float(reference))
    )
    decibels = np.maximum(decibels, float(floor_db))
    if decibels.ndim == 0:
        return float(decibels)
    return decibels


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Convert a positive duration to its nearest positive sample count."""

    validated_rate = validate_sample_rate(sample_rate)
    if isinstance(seconds, bool) or not isinstance(seconds, Real):
        raise TypeError("seconds must be a positive real number")
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and greater than zero")
    count = int(round(float(seconds) * validated_rate))
    if count <= 0:
        raise ValueError("duration is shorter than one sample")
    return count


def to_json_safe(value: Any) -> Any:
    """Recursively convert SDK results into strict JSON-compatible values.

    Dataclasses are traversed without ``dataclasses.asdict`` so large NumPy
    audio arrays are not deep-copied before conversion.
    """

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def _nested_section(
    config: Mapping[str, Any], *path_parts: str
) -> Mapping[str, Any]:
    current = config
    traversed: list[str] = []
    for part in path_parts:
        traversed.append(part)
        try:
            value = current[part]
        except KeyError as exc:
            raise ConfigError(
                f"Missing configuration section: {'.'.join(traversed)}"
            ) from exc
        if not isinstance(value, Mapping):
            raise ConfigError(
                f"Configuration section '{'.'.join(traversed)}' must be a mapping"
            )
        current = value
    return current


def _required_integer(
    mapping: Mapping[str, Any], key: str, *, path: str
) -> int:
    try:
        value = mapping[key]
    except KeyError as exc:
        raise ConfigError(f"Missing configuration value: {path}") from exc
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ConfigError(f"{path} must be an integer")
    return int(value)


def _required_real(
    mapping: Mapping[str, Any], key: str, *, path: str
) -> float:
    try:
        value = mapping[key]
    except KeyError as exc:
        raise ConfigError(f"Missing configuration value: {path}") from exc
    return _coerce_real(value, path=path)


def _coerce_real(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigError(f"{path} must be a real number")
    converted = float(value)
    if not np.isfinite(converted):
        raise ConfigError(f"{path} must be finite")
    return converted


def _required_path_real(
    config: Mapping[str, Any], path_parts: tuple[str, ...]
) -> float:
    parent = _nested_section(config, *path_parts[:-1])
    return _required_real(
        parent,
        path_parts[-1],
        path=".".join(path_parts),
    )
