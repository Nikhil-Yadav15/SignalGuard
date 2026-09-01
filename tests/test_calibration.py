"""Tests for deterministic, reviewable threshold calibration."""

from __future__ import annotations

import pytest

from src.calibration import CalibrationExample, CalibrationLabel, calibrate_likely_synthetic_threshold, score_labelled_audio


def test_calibration_selects_perfectly_separating_threshold() -> None:
    report = calibrate_likely_synthetic_threshold(
        (
            CalibrationExample("n1", CalibrationLabel.NATURAL, 10.0),
            CalibrationExample("n2", CalibrationLabel.NATURAL, 25.0),
            CalibrationExample("s1", CalibrationLabel.SYNTHETIC, 65.0),
            CalibrationExample("s2", CalibrationLabel.SYNTHETIC, 80.0),
        )
    )
    assert report.recommended_likely_synthetic_threshold == 65.0
    assert report.metrics.f1 == 1.0
    assert report.metrics.false_positive == report.metrics.false_negative == 0
    assert report.to_dict()["metrics"]["precision"] == 1.0


def test_calibration_rejects_incomplete_or_invalid_labels() -> None:
    with pytest.raises(ValueError, match="at least two"):
        calibrate_likely_synthetic_threshold((CalibrationExample("n", CalibrationLabel.NATURAL, 20.0),))
    with pytest.raises(ValueError, match="natural and one synthetic"):
        calibrate_likely_synthetic_threshold(
            (CalibrationExample("n1", CalibrationLabel.NATURAL, 20.0), CalibrationExample("n2", CalibrationLabel.NATURAL, 30.0))
        )


def test_directory_calibration_fails_openly_when_data_is_missing(tmp_path) -> None:
    natural = tmp_path / "natural"; synthetic = tmp_path / "synthetic"; natural.mkdir(); synthetic.mkdir()
    with pytest.raises(ValueError, match="No supported audio files"):
        score_labelled_audio(natural, synthetic)
