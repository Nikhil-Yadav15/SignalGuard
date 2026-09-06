"""End-to-end deterministic SignalGuard decision pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .detectors import (
    analyze_bispectrum,
    analyze_breath,
    analyze_decay,
    analyze_f0,
    analyze_harmonics,
    analyze_lpc,
    analyze_modulation,
    analyze_phase,
    analyze_spectral,
    analyze_temporal,
    detect_corruption,
)
from .detectors._common import AnalysisResult
from .detectors.corruption_detection import CorruptionReport, Severity
from .detectors.f0_analysis import F0AnalysisResult, F0Settings
from .preprocessing import (
    AudioSource,
    PreprocessedAudio,
    PreprocessingSettings,
    preprocess_audio,
    preprocess_samples,
    segment_audio,
)
from .quality import QualityValidation, validate_restoration
from .restoration import (
    highpass_rumble,
    lowpass_hiss,
    notch_hum,
    repair_clipping,
    repair_dropouts,
    repair_impulses,
    spectral_subtract,
)
from .scoring import (
    EvidenceDecision,
    SyntheticEvidenceScore,
    aggregate_segment_scores,
    combine_evidence_scores,
    score_synthetic_evidence,
)
from .utils import (
    get_config_section,
    load_config,
    to_json_safe,
    validate_project_config,
)


class PipelineDecision(str, Enum):
    PASS = "PASS"
    PASS_RESTORED = "PASS_RESTORED"
    REJECT_LIKELY_SYNTHETIC = "REJECT_LIKELY_SYNTHETIC"
    REJECT_UNRECOVERABLE = "REJECT_UNRECOVERABLE"


ForensicResult = F0AnalysisResult | AnalysisResult


@dataclass(frozen=True, slots=True)
class PipelineResult:
    decision: PipelineDecision
    audio: PreprocessedAudio
    evidence: SyntheticEvidenceScore
    segment_evidence: tuple[SyntheticEvidenceScore, ...]
    forensic_results: dict[str, ForensicResult]
    corruption_before: CorruptionReport
    corruption_after: CorruptionReport | None
    restored_samples: tuple[float, ...] | None
    applied_restorations: tuple[str, ...]
    quality: QualityValidation | None

    def to_dict(self) -> dict[str, Any]:
        """Return the complete result as strict JSON-safe primitives."""

        return to_json_safe(self)


class SignalGuardPipeline:
    """Configuration-isolated orchestrator for analysis and restoration."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        loaded = load_config() if config is None else deepcopy(dict(config))
        validate_project_config(loaded)
        self.config = loaded
        self.preprocessing_settings = PreprocessingSettings.from_config(loaded)
        self.f0_settings = F0Settings.from_config(loaded)

    def analyze_file(self, source: AudioSource) -> PipelineResult:
        audio = preprocess_audio(
            source,
            settings=self.preprocessing_settings,
        )
        return self._run(audio)

    def analyze_samples(
        self,
        samples: ArrayLike,
        sample_rate: int,
        *,
        channel_axis: int = -1,
    ) -> PipelineResult:
        audio = preprocess_samples(
            samples,
            sample_rate,
            settings=self.preprocessing_settings,
            channel_axis=channel_axis,
        )
        return self._run(audio)

    def _run(self, audio: PreprocessedAudio) -> PipelineResult:
        samples = audio.samples
        rate = audio.sample_rate
        segments = segment_audio(
            samples,
            rate,
            settings=self.preprocessing_settings,
        )
        segment_results = [
            self._analyze_forensic_channels(segment.samples, rate)
            for segment in segments
        ]
        segment_scores = tuple(
            score_synthetic_evidence(result, config=self.config)
            for result in segment_results
        )
        segment_aggregate = aggregate_segment_scores(
            segment_scores,
            config=self.config,
        )

        f0_segments = [
            result["f0"]
            for result in segment_results
            if isinstance(result["f0"], F0AnalysisResult)
        ]
        temporal = analyze_temporal(
            [segment.samples for segment in segments],
            rate,
            f0_results=f0_segments,
            config=self.config,
        )
        temporal_score = score_synthetic_evidence(
            {"temporal": temporal},
            config=self.config,
        )
        domain_weights = get_config_section(
            get_config_section(self.config, "forensics"),
            "scoring_weights",
        )
        non_temporal_weight = sum(
            float(weight)
            for name, weight in domain_weights.items()
            if name != "temporal"
        )
        evidence = combine_evidence_scores(
            (
                (segment_aggregate, non_temporal_weight),
                (temporal_score, float(domain_weights["temporal"])),
            ),
            config=self.config,
        )

        # Preserve the familiar clip-level detector view for callers while the
        # decision itself uses boundary-safe segment aggregation.
        if len(segment_results) == 1:
            forensic_results = dict(segment_results[0])
        else:
            forensic_results = self._analyze_forensic_channels(samples, rate)
        forensic_results["temporal"] = temporal

        corruption_before = detect_corruption(
            samples,
            rate,
            config=self.config,
        )
        if evidence.decision is EvidenceDecision.LIKELY_SYNTHETIC:
            return PipelineResult(
                PipelineDecision.REJECT_LIKELY_SYNTHETIC,
                audio,
                evidence,
                segment_scores,
                forensic_results,
                corruption_before,
                None,
                None,
                (),
                None,
            )
        if not corruption_before.detected_kinds:
            return PipelineResult(
                PipelineDecision.PASS,
                audio,
                evidence,
                segment_scores,
                forensic_results,
                corruption_before,
                None,
                None,
                (),
                None,
            )

        restored, applied = self._restore(
            samples,
            rate,
            corruption_before,
        )
        quality = validate_restoration(
            samples,
            restored,
            rate,
            before_report=corruption_before,
            config=self.config,
        )
        if quality.accepted:
            decision = PipelineDecision.PASS_RESTORED
            restored_output: tuple[float, ...] | None = tuple(
                float(value) for value in np.asarray(restored).flat
            )
            applied_output = tuple(applied)
            quality_after = quality.after
        else:
            # Restoration was rejected by the quality gate.
            # Only declare the audio REJECT_UNRECOVERABLE if the original recording
            # suffered from severe distortions (e.g. severe clipping, SNR < 12 dB, major dropouts).
            # If the detected corruption was minor/moderate or a false positive,
            # we safely bypass restoration and PASS the original clean audio.
            is_severe = any(
                d.severity in (Severity.SEVERE, Severity.HIGH)
                for d in corruption_before.detections
                if d.detected
            )
            if is_severe:
                decision = PipelineDecision.REJECT_UNRECOVERABLE
                restored_output = tuple(
                    float(value) for value in np.asarray(restored).flat
                )
                applied_output = tuple(applied)
                quality_after = quality.after
            else:
                decision = PipelineDecision.PASS
                restored_output = None
                applied_output = ()
                quality_after = None

        return PipelineResult(
            decision,
            audio,
            evidence,
            segment_scores,
            forensic_results,
            corruption_before,
            quality_after,
            restored_output,
            applied_output,
            quality,
        )

    def _analyze_forensic_channels(
        self,
        samples: ArrayLike,
        sample_rate: int,
    ) -> dict[str, ForensicResult]:
        f0 = analyze_f0(
            samples,
            sample_rate,
            settings=self.f0_settings,
        )
        return {
            "f0": f0,
            "harmonic": analyze_harmonics(
                samples,
                sample_rate,
                f0_result=f0,
                config=self.config,
            ),
            "spectral": analyze_spectral(
                samples,
                sample_rate,
                config=self.config,
            ),
            "phase": analyze_phase(
                samples,
                sample_rate,
                config=self.config,
            ),
            "lpc": analyze_lpc(
                samples,
                sample_rate,
                config=self.config,
            ),
            "bispectrum": analyze_bispectrum(
                samples,
                sample_rate,
                config=self.config,
            ),
            "modulation": analyze_modulation(
                samples,
                sample_rate,
                config=self.config,
            ),
            "breath": analyze_breath(
                samples,
                sample_rate,
                config=self.config,
            ),
            "decay": analyze_decay(
                samples,
                sample_rate,
                config=self.config,
            ),
        }

    def _restore(
        self,
        samples: ArrayLike,
        sample_rate: int,
        report: CorruptionReport,
    ) -> tuple[ArrayLike, list[str]]:
        restored = np.asarray(samples, dtype=np.float32).copy()
        applied: list[str] = []
        restoration = get_config_section(self.config, "restoration")
        corruption = get_config_section(self.config, "corruption_detection")
        detected = set(report.detected_kinds)

        if "broadband_noise" in detected:
            restored = spectral_subtract(
                restored,
                sample_rate,
                restoration["noise_reduction"],
            )
            applied.append("broadband_noise")
        if "hum" in detected:
            restored = notch_hum(
                restored,
                sample_rate,
                mains_hz=float(corruption["hum"]["mains_frequency_hz"]),
                harmonic_count=(
                    int(corruption["hum"]["harmonic_count"])
                    if bool(restoration["notch_filter"]["apply_harmonics"])
                    else 1
                ),
                quality_factor=float(
                    restoration["notch_filter"]["quality_factor"]
                ),
            )
            applied.append("hum")
        if "impulse" in detected:
            restored = repair_impulses(
                restored,
                sample_rate,
                kernel_size=int(
                    restoration["impulse_repair"]["median_kernel_size"]
                ),
                context_samples=int(
                    restoration["impulse_repair"]["interpolation_context_samples"]
                ),
                threshold_multiplier=float(
                    corruption["impulse"]["median_absolute_deviation_multiplier"]
                ),
                minimum_absolute_amplitude=float(
                    corruption["impulse"]["minimum_absolute_amplitude"]
                ),
                minimum_residual_amplitude=float(
                    corruption["impulse"]["minimum_residual_amplitude"]
                ),
            )
            applied.append("impulse")
        if "dropout" in detected:
            restored = repair_dropouts(
                restored,
                sample_rate,
                near_zero_amplitude=float(
                    corruption["dropout"]["near_zero_amplitude"]
                ),
                minimum_duration_ms=float(
                    corruption["dropout"]["minimum_duration_ms"]
                ),
                maximum_duration_ms=float(
                    corruption["dropout"]["maximum_restorable_duration_ms"]
                ),
            )
            applied.append("dropout")
        if "rumble" in detected:
            restored = highpass_rumble(
                restored,
                sample_rate,
                cutoff_hz=float(restoration["rumble_filter"]["cutoff_hz"]),
                order=int(restoration["rumble_filter"]["order"]),
            )
            applied.append("rumble")
        if "hiss" in detected:
            restored = lowpass_hiss(
                restored,
                sample_rate,
                cutoff_hz=float(restoration["hiss_filter"]["cutoff_hz"]),
                order=int(restoration["hiss_filter"]["order"]),
            )
            applied.append("hiss")
        if (
            "clipping" in detected
            and float(report.metrics["clipped_sample_ratio"] or 0.0)
            <= float(restoration["declipping"]["maximum_repair_ratio"])
        ):
            restored = repair_clipping(
                restored,
                sample_rate,
                sample_level=float(corruption["clipping"]["sample_level"]),
                flat_top_min_samples=int(
                    corruption["clipping"]["flat_top_min_samples"]
                ),
                flat_top_tolerance=float(
                    corruption["clipping"]["flat_top_tolerance"]
                ),
                context_samples=int(restoration["declipping"]["context_samples"]),
            )
            applied.append("clipping")
        return restored, applied
