"""Interpretable SignalGuard detector package."""

from .bispectrum_analysis import analyze_bispectrum
from .breath_analysis import analyze_breath
from .corruption_detection import (
    CorruptionReport,
    DetectedCorruption,
    Severity,
    detect_corruption,
)
from .decay_analysis import analyze_decay
from .f0_analysis import (
    F0AnalysisResult,
    F0AnalysisStatus,
    F0Metrics,
    F0RuleEvaluation,
    F0Settings,
    F0Track,
    analyze_f0,
    calculate_f0_differences,
)
from .harmonic_analysis import analyze_harmonics
from .lpc_analysis import analyze_lpc
from .modulation_analysis import analyze_modulation
from .phase_analysis import analyze_phase
from .spectral_analysis import analyze_spectral
from .temporal_analysis import analyze_temporal

__all__ = [
    "CorruptionReport",
    "DetectedCorruption",
    "F0AnalysisResult",
    "F0AnalysisStatus",
    "F0Metrics",
    "F0RuleEvaluation",
    "F0Settings",
    "F0Track",
    "Severity",
    "analyze_bispectrum",
    "analyze_breath",
    "analyze_decay",
    "analyze_f0",
    "analyze_harmonics",
    "analyze_lpc",
    "analyze_modulation",
    "analyze_phase",
    "analyze_spectral",
    "analyze_temporal",
    "calculate_f0_differences",
    "detect_corruption",
]
