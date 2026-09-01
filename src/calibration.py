"""Deterministic calibration utilities for labelled SignalGuard recordings.

The toolkit selects a decision threshold from held-out evidence scores.  It
does not overwrite ``config.yaml``: calibration is a policy choice that needs
review, provenance, and a held-out validation set.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import SignalGuardPipeline
from .utils import to_json_safe


class CalibrationLabel(str, Enum):
    NATURAL = "natural"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    """One scored recording and its ground-truth provenance label."""

    identifier: str
    label: CalibrationLabel
    score: float

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("identifier must not be empty")
        if not np.isfinite(self.score) or not 0.0 <= self.score <= 100.0:
            raise ValueError("score must be finite and between 0 and 100")


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    specificity: float
    f1: float


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    recommended_likely_synthetic_threshold: float
    metrics: ThresholdMetrics
    example_count: int
    natural_count: int
    synthetic_count: int
    selection_rule: str
    def to_dict(self) -> dict[str, Any]: return to_json_safe(self)


def calibrate_likely_synthetic_threshold(examples: Sequence[CalibrationExample]) -> CalibrationReport:
    """Select the F1-optimal threshold with deterministic conservative ties.

    A score equal to the threshold is classified synthetic, matching the
    runtime decision boundary. Ties choose the higher threshold, which reduces
    false positives when F1 is otherwise identical.
    """
    if len(examples) < 2:
        raise ValueError("at least two labelled examples are required")
    labels={example.label for example in examples}
    if labels != {CalibrationLabel.NATURAL, CalibrationLabel.SYNTHETIC}:
        raise ValueError("calibration requires at least one natural and one synthetic example")
    values=sorted({float(example.score) for example in examples})
    candidates=sorted(set([0.0, 100.0, *values]), reverse=True)
    metrics=[_metrics(examples, candidate) for candidate in candidates]
    best=max(metrics, key=lambda item: (item.f1, item.precision, item.specificity, item.threshold))
    return CalibrationReport(best.threshold,best,len(examples),sum(item.label is CalibrationLabel.NATURAL for item in examples),sum(item.label is CalibrationLabel.SYNTHETIC for item in examples),"Maximum F1; ties prefer precision, specificity, then the higher threshold.")


def score_labelled_audio(natural_directory: str | Path, synthetic_directory: str | Path, *, pipeline: SignalGuardPipeline | None = None) -> tuple[CalibrationExample, ...]:
    """Run the pipeline on labelled directory trees and return raw evidence scores.

    Files are never uploaded or modified. Decode failures are reported with the
    file path rather than silently removing evidence from a calibration set.
    """
    active=pipeline or SignalGuardPipeline()
    pairs=((Path(natural_directory),CalibrationLabel.NATURAL),(Path(synthetic_directory),CalibrationLabel.SYNTHETIC))
    result=[]
    extensions={".wav", ".flac", ".ogg", ".mp3"}
    for directory,label in pairs:
        if not directory.is_dir(): raise FileNotFoundError(f"Calibration directory does not exist: {directory}")
        files=sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
        if not files: raise ValueError(f"No supported audio files found in calibration directory: {directory}")
        for path in files:
            try: analysis=active.analyze_file(str(path))
            except Exception as exc: raise ValueError(f"Unable to score calibration file: {path}") from exc
            result.append(CalibrationExample(str(path),label,analysis.evidence.score))
    return tuple(result)


def calibrate_from_directories(natural_directory: str | Path, synthetic_directory: str | Path, *, pipeline: SignalGuardPipeline | None = None) -> CalibrationReport:
    """Score labelled recordings and return a reviewable threshold report."""
    return calibrate_likely_synthetic_threshold(score_labelled_audio(natural_directory,synthetic_directory,pipeline=pipeline))


def _metrics(examples: Iterable[CalibrationExample], threshold: float) -> ThresholdMetrics:
    tp=fp=tn=fn=0
    for example in examples:
        predicted=example.score>=threshold; actual=example.label is CalibrationLabel.SYNTHETIC
        if predicted and actual: tp+=1
        elif predicted: fp+=1
        elif actual: fn+=1
        else: tn+=1
    precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0; specificity=tn/(tn+fp) if tn+fp else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return ThresholdMetrics(float(threshold),tp,fp,tn,fn,float(precision),float(recall),float(specificity),float(f1))
