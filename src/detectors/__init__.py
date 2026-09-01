"""Interpretable SignalGuard detector package."""

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
from .spectral_analysis import analyze_spectral
from .phase_analysis import analyze_phase
from .temporal_analysis import analyze_temporal
from .corruption_detection import CorruptionReport, DetectedCorruption, Severity, detect_corruption

__all__ = [
    "F0AnalysisResult",
    "F0AnalysisStatus",
    "F0Metrics",
    "F0RuleEvaluation",
    "F0Settings",
    "F0Track",
    "analyze_f0",
    "calculate_f0_differences",
    "analyze_harmonics",
    "analyze_spectral",
    "analyze_phase",
    "analyze_temporal",
    "CorruptionReport",
    "DetectedCorruption",
    "Severity",
    "detect_corruption",
]
