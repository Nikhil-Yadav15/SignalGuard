"""Tests for dataset generator, evaluation runner, and CLI integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.cli import main as cli_main
from src.dataset_generator import generate_benchmark_dataset
from src.evaluate import evaluate_pipeline_on_datasets


def test_dataset_generator_creates_valid_files(tmp_path: Path) -> None:
    counts = generate_benchmark_dataset(
        output_dir=tmp_path,
        samples_per_class=2,
        sample_rate=16_000,
    )
    assert counts["clean"] == 2
    assert counts["synthetic"] == 2
    assert counts["corrupted"] == 2

    for folder in ("clean", "synthetic", "corrupted"):
        wav_files = list((tmp_path / folder).glob("*.wav"))
        assert len(wav_files) == 2
        for wav_path in wav_files:
            data, sr = sf.read(wav_path)
            assert sr == 16_000
            assert len(data) > 0
            assert np.all(np.isfinite(data))


def test_evaluate_pipeline_on_generated_datasets(tmp_path: Path) -> None:
    generate_benchmark_dataset(
        output_dir=tmp_path,
        samples_per_class=2,
        sample_rate=16_000,
    )
    results = evaluate_pipeline_on_datasets(
        clean_dir=tmp_path / "clean",
        synthetic_dir=tmp_path / "synthetic",
        corrupted_dir=tmp_path / "corrupted",
    )
    assert "Clean Human" in results
    assert "AI-Generated" in results
    assert "Corrupted Human" in results
    assert results["Clean Human"].sample_count == 2
    assert results["AI-Generated"].sample_count == 2


def test_cli_analyze_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    generate_benchmark_dataset(
        output_dir=tmp_path,
        samples_per_class=1,
        sample_rate=16_000,
    )
    sample_file = next((tmp_path / "clean").glob("*.wav"))

    # Test summary view
    cli_main(["analyze", str(sample_file)])
    out = capsys.readouterr().out
    assert "SIGNALGUARD ANALYSIS" in out
    assert "Decision" in out

    # Test JSON output
    cli_main(["analyze", str(sample_file), "--json"])
    json_out = capsys.readouterr().out
    parsed = json.loads(json_out)
    assert "decision" in parsed
    assert "evidence" in parsed
