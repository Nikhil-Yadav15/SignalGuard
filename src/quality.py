"""Objective post-restoration quality gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .detectors._common import mono_audio
from .detectors.corruption_detection import CorruptionReport, detect_corruption
from .utils import get_config_section, load_config, to_json_safe


@dataclass(frozen=True, slots=True)
class QualityCheck:
    """One transparent post-restoration acceptance check."""

    code: str
    passed: bool
    before_value: float | None
    after_value: float | None
    required_value: float | None
    explanation: str


@dataclass(frozen=True, slots=True)
class QualityValidation:
    accepted: bool
    rms_change_db: float
    checks: tuple[QualityCheck, ...]
    reasons: tuple[str, ...]
    before: CorruptionReport
    after: CorruptionReport

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe(self)


def validate_restoration(
    before_samples: ArrayLike,
    after_samples: ArrayLike,
    sample_rate: int,
    *,
    before_report: CorruptionReport | None = None,
    config: Mapping[str, Any] | None = None,
) -> QualityValidation:
    """Accept restoration only when configured objective checks succeed."""

    root = load_config() if config is None else config
    settings = get_config_section(root, "quality_validation")
    success = get_config_section(settings, "success_by_corruption")
    before, rate = mono_audio(before_samples, sample_rate)
    after, _ = mono_audio(after_samples, rate)
    if before.shape != after.shape:
        raise ValueError("restored audio must preserve the input sample count")

    before_result = before_report or detect_corruption(before, rate, config=root)
    if before_result.sample_rate_hz != rate:
        raise ValueError("before_report sample rate does not match the audio")
    after_result = detect_corruption(after, rate, config=root)

    before_rms = _rms(before)
    after_rms = _rms(after)
    rms_change = float(
        20.0
        * np.log10(
            max(after_rms, np.finfo(np.float64).tiny)
            / max(before_rms, np.finfo(np.float64).tiny)
        )
    )
    checks: list[QualityCheck] = []

    maximum_rms_change = float(settings["maximum_rms_change_db"])
    checks.append(
        QualityCheck(
            "rms_change",
            abs(rms_change) <= maximum_rms_change,
            before_rms,
            after_rms,
            maximum_rms_change,
            "Absolute RMS change must remain within the configured dB limit.",
        )
    )

    post_clipping = _metric(after_result, "clipped_sample_ratio")
    maximum_clipping = float(settings["maximum_clipping_ratio"])
    checks.append(
        QualityCheck(
            "post_clipping",
            post_clipping is not None and post_clipping <= maximum_clipping,
            _metric(before_result, "clipped_sample_ratio"),
            post_clipping,
            maximum_clipping,
            "Post-restoration clipping must not exceed the configured ratio.",
        )
    )

    if bool(settings["require_corruption_reduction"]):
        for kind in before_result.detected_kinds:
            checks.append(
                _corruption_reduction_check(
                    kind,
                    before_result,
                    after_result,
                    settings,
                    success,
                )
            )

    if bool(settings["reject_new_corruptions"]):
        introduced = sorted(
            set(after_result.detected_kinds) - set(before_result.detected_kinds)
        )
        checks.append(
            QualityCheck(
                "no_new_corruptions",
                not introduced,
                float(len(before_result.detected_kinds)),
                float(len(after_result.detected_kinds)),
                0.0,
                (
                    "No new corruption indicators were introduced."
                    if not introduced
                    else "New corruption indicators: " + ", ".join(introduced)
                ),
            )
        )

    failed_reasons = tuple(
        check.explanation for check in checks if not check.passed
    )
    return QualityValidation(
        accepted=not failed_reasons,
        rms_change_db=rms_change,
        checks=tuple(checks),
        reasons=failed_reasons,
        before=before_result,
        after=after_result,
    )


def _corruption_reduction_check(
    kind: str,
    before: CorruptionReport,
    after: CorruptionReport,
    quality_settings: Mapping[str, Any],
    success_settings: Mapping[str, Any],
) -> QualityCheck:
    kind_settings = get_config_section(success_settings, kind)
    if kind == "broadband_noise":
        before_value = _metric(before, "estimated_snr_db")
        after_value = _metric(after, "estimated_snr_db")
        minimum_improvement = max(
            float(quality_settings["minimum_snr_improvement_db"]),
            float(kind_settings["minimum_snr_improvement_db"]),
        )
        minimum_post_snr = max(
            float(quality_settings["minimum_acceptable_snr_db"]),
            float(kind_settings["minimum_post_snr_db"]),
        )
        improvement = (
            None
            if before_value is None or after_value is None
            else after_value - before_value
        )
        passed = (
            improvement is not None
            and improvement >= minimum_improvement
            and after_value is not None
            and after_value >= minimum_post_snr
        )
        return QualityCheck(
            "broadband_noise_reduction",
            passed,
            before_value,
            after_value,
            minimum_improvement,
            (
                "Broadband-noise restoration met SNR improvement and post-SNR limits."
                if passed
                else "Broadband-noise restoration did not meet the configured SNR improvement and post-SNR limits."
            ),
        )

    metric_by_kind = {
        "hum": ("hum_prominence_db", "minimum_prominence_reduction_db", "difference"),
        "impulse": ("impulse_events_per_second", "minimum_event_reduction_ratio", "ratio"),
        "dropout": ("total_dropout_ratio", "minimum_duration_reduction_ratio", "ratio"),
        "rumble": ("low_band_energy_ratio", "minimum_band_energy_reduction_ratio", "ratio"),
        "hiss": ("high_band_energy_ratio", "minimum_band_energy_reduction_ratio", "ratio"),
        "clipping": ("clipped_sample_ratio", "minimum_clipping_reduction_ratio", "ratio"),
    }
    metric_name, threshold_name, reduction_type = metric_by_kind[kind]
    before_value = _metric(before, metric_name)
    after_value = _metric(after, metric_name)
    required = float(kind_settings[threshold_name])
    if before_value is None:
        reduction = None
    elif after_value is None:
        reduction = float("inf") if reduction_type == "difference" else 1.0
    elif reduction_type == "difference":
        reduction = before_value - after_value
    elif before_value <= np.finfo(np.float64).tiny:
        reduction = 0.0
    else:
        reduction = (before_value - after_value) / before_value
    passed = reduction is not None and reduction >= required
    return QualityCheck(
        f"{kind}_reduction",
        passed,
        before_value,
        after_value,
        required,
        (
            f"{kind} was reduced by the configured objective amount."
            if passed
            else f"{kind} did not improve by the configured objective amount."
        ),
    )


def _metric(report: CorruptionReport, name: str) -> float | None:
    value = report.metrics.get(name)
    return None if value is None else float(value)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
