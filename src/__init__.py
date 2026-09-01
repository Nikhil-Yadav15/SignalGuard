"""SignalGuard deterministic audio forensics and restoration SDK."""

from .preprocessing import (
    AudioLoadError,
    AudioSegment,
    PreprocessedAudio,
    PreprocessingSettings,
    preprocess_audio,
    preprocess_samples,
    segment_audio,
)
from .utils import ConfigError, load_config, validate_project_config
from .pipeline import PipelineDecision, PipelineResult, SignalGuardPipeline
from .scoring import EvidenceDecision, SyntheticEvidenceScore
from .calibration import (
    CalibrationExample,
    CalibrationLabel,
    CalibrationReport,
    calibrate_from_directories,
    calibrate_likely_synthetic_threshold,
    score_labelled_audio,
)

__all__ = [
    "AudioLoadError",
    "AudioSegment",
    "ConfigError",
    "PreprocessedAudio",
    "PreprocessingSettings",
    "load_config",
    "preprocess_audio",
    "preprocess_samples",
    "segment_audio",
    "validate_project_config",
    "EvidenceDecision",
    "PipelineDecision",
    "PipelineResult",
    "SignalGuardPipeline",
    "SyntheticEvidenceScore",
    "CalibrationExample",
    "CalibrationLabel",
    "CalibrationReport",
    "calibrate_from_directories",
    "calibrate_likely_synthetic_threshold",
    "score_labelled_audio",
]
