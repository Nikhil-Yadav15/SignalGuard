"""Tests for configuration and numerical utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.preprocessing import PreprocessingSettings
from src.utils import (
    ConfigError,
    load_config,
    power_to_db,
    rms,
    seconds_to_samples,
    validate_audio_array,
    validate_project_config,
)


def test_load_config_parses_mapping_and_typed_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "preprocessing:\n  target_sample_rate: 16000\n  enabled: true\n",
        encoding="utf-8",
    )

    config = load_config(config_path, validate=False)

    assert config["preprocessing"]["target_sample_rate"] == 16_000
    assert config["preprocessing"]["enabled"] is True


def test_default_config_lookup_is_independent_of_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config()
    settings = PreprocessingSettings.from_config(config)

    assert settings.target_sample_rate == 16_000
    assert settings.segment_duration_seconds == 3.0


def test_default_project_config_passes_cross_section_validation() -> None:
    validate_project_config(load_config())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: config["forensics"]["scoring_weights"].__setitem__(
                "f0", 24.0
            ),
            "weights must total 100",
        ),
        (
            lambda config: config["forensics"]["decision_thresholds"].__setitem__(
                "review_min", 20.0
            ),
            "strictly increasing",
        ),
        (
            lambda config: config["restoration"]["hiss_filter"].__setitem__(
                "cutoff_hz", 8_000.0
            ),
            "below Nyquist",
        ),
        (
            lambda config: config["restoration"]["impulse_repair"].__setitem__(
                "median_kernel_size", 4
            ),
            "positive odd integer",
        ),
        (
            lambda config: config["corruption_detection"]["clipping"].__setitem__(
                "moderate_ratio", 0.0001
            ),
            "clipping ratios must be strictly increasing",
        ),
        (
            lambda config: config["quality_validation"][
                "success_by_corruption"
            ].pop("hum"),
            "Missing configuration section: hum",
        ),
        (
            lambda config: config["quality_validation"][
                "success_by_corruption"
            ]["impulse"].__setitem__("minimum_event_reduction_ratio", 2.0),
            "reduction ratios must not exceed 1",
        ),
    ],
)
def test_project_config_rejects_cross_section_invariant_violations(
    mutation: object, message: str
) -> None:
    config = deepcopy(load_config())
    mutation(config)  # type: ignore[operator]

    with pytest.raises(ConfigError, match=message):
        validate_project_config(config)


def test_load_config_rejects_malformed_or_non_mapping_yaml(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("key: [unterminated", encoding="utf-8")
    non_mapping = tmp_path / "list.yaml"
    non_mapping.write_text("- first\n- second\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(malformed)
    with pytest.raises(ConfigError, match="top-level mapping"):
        load_config(non_mapping)


def test_load_config_propagates_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_runs_full_validation_by_default(tmp_path: Path) -> None:
    invalid = deepcopy(load_config())
    invalid["forensics"]["decision_thresholds"]["review_min"] = 20.0
    config_path = tmp_path / "invalid-project.yaml"
    config_path.write_text(yaml.safe_dump(invalid), encoding="utf-8")

    with pytest.raises(ConfigError, match="strictly increasing"):
        load_config(config_path)


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([], dtype=np.float32),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([True, False]),
        np.array([1 + 1j]),
        np.array(["audio"]),
        np.zeros((1, 1, 1)),
    ],
)
def test_validate_audio_array_rejects_invalid_values(invalid: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_audio_array(invalid)


def test_rms_and_power_to_db_match_analytic_values() -> None:
    sine = _unit_sine(1_000, 16_000)

    assert rms(sine) == pytest.approx(1 / np.sqrt(2), rel=1e-6)
    assert power_to_db(1.0) == pytest.approx(0.0)
    assert power_to_db(0.25) == pytest.approx(-6.0205999)
    assert power_to_db(0.0, floor_db=-100.0) == pytest.approx(-100.0)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("reference", np.nan),
        ("reference", np.inf),
        ("floor_db", np.nan),
        ("floor_db", np.inf),
        ("floor_db", 1.0),
    ],
)
def test_power_to_db_rejects_nonfinite_or_invalid_parameters(
    keyword: str, value: float
) -> None:
    arguments = {keyword: value}
    with pytest.raises(ValueError):
        power_to_db(1.0, **arguments)


def test_seconds_to_samples_validates_duration() -> None:
    assert seconds_to_samples(3.0, 16_000) == 48_000
    with pytest.raises(ValueError):
        seconds_to_samples(0.0, 16_000)
    with pytest.raises(ValueError):
        seconds_to_samples(0.000001, 16_000)


def _unit_sine(frequency_hz: float, sample_rate: int) -> np.ndarray:
    times = np.arange(sample_rate, dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * frequency_hz * times)
