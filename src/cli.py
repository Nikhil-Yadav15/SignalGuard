"""Unified Command-Line Interface (CLI) for SignalGuard.

Supports dataset generation, full pipeline evaluation, threshold calibration,
and single-file forensic inspection.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

with contextlib.suppress(ImportError):
    import numba  # noqa: F401

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset_generator import generate_benchmark_dataset
from src.evaluate import (
    evaluate_pipeline_on_datasets,
    print_evaluation_report,
    run_calibration,
)
from src.pipeline import SignalGuardPipeline


def _cmd_analyze(args: argparse.Namespace) -> None:
    pipeline = SignalGuardPipeline()
    file_path = Path(args.file)
    if not file_path.is_file():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    result = pipeline.analyze_file(str(file_path))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print("=" * 60)
    print(f" SIGNALGUARD ANALYSIS: {file_path.name}")
    print("=" * 60)
    print(f"  Decision                 : {result.decision.value}")
    print(f"  Synthetic Evidence Score : {result.evidence.score:.1f}/100 ({result.evidence.decision.value})")
    print(f"  Sample Rate              : {result.audio.sample_rate:,} Hz")
    print(f"  Duration                 : {result.audio.duration_seconds:.2f} s")

    if result.evidence.explanations:
        print("\nForensic Anomaly Explanations:")
        for exp in result.evidence.explanations:
            print(f"  • {exp}")

    corruptions = result.corruption_before.detected_kinds
    print(f"\nCorruptions Detected       : {', '.join(corruptions) if corruptions else 'None (Clean)'}")
    if result.applied_restorations:
        print(f"Applied DSP Restorations   : {', '.join(result.applied_restorations)}")
    if result.quality:
        status = "Accepted" if result.quality.accepted else "Rejected"
        print(f"Quality Gate Outcome       : {status}")


def _cmd_generate_data(args: argparse.Namespace) -> None:
    print(f"Generating benchmark dataset ({args.count} files/class) into '{args.output_dir}'...")
    counts = generate_benchmark_dataset(
        output_dir=args.output_dir,
        samples_per_class=args.count,
        sample_rate=args.sample_rate,
    )
    print("Dataset generation completed:")
    for k, v in counts.items():
        print(f"  • {args.output_dir}/{k}: {v} WAV files")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    results = evaluate_pipeline_on_datasets(
        clean_dir=args.clean_dir,
        synthetic_dir=args.synthetic_dir,
        corrupted_dir=args.corrupted_dir,
    )
    print_evaluation_report(results)
    if not args.skip_calibration:
        run_calibration(clean_dir=args.clean_dir, synthetic_dir=args.synthetic_dir)


def _cmd_calibrate(args: argparse.Namespace) -> None:
    run_calibration(clean_dir=args.natural_dir, synthetic_dir=args.synthetic_dir)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="signalguard",
        description="SignalGuard · Deterministic Audio Forensics and Restoration SDK",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze an audio file")
    p_analyze.add_argument("file", help="Path to audio file (.wav, .flac, .ogg, .mp3)")
    p_analyze.add_argument("--json", action="store_true", help="Output full analysis result as JSON")
    p_analyze.set_defaults(func=_cmd_analyze)

    # generate-data
    p_gen = subparsers.add_parser("generate-data", help="Generate benchmark audio dataset")
    p_gen.add_argument("--output-dir", default="data", help="Target output directory (default: data)")
    p_gen.add_argument("--count", type=int, default=10, help="Samples per class (default: 10)")
    p_gen.add_argument("--sample-rate", type=int, default=16_000, help="Sample rate (default: 16000)")
    p_gen.set_defaults(func=_cmd_generate_data)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run comprehensive pipeline evaluation")
    p_eval.add_argument("--clean-dir", default="data/clean", help="Clean audio directory")
    p_eval.add_argument("--synthetic-dir", default="data/synthetic", help="Synthetic audio directory")
    p_eval.add_argument("--corrupted-dir", default="data/corrupted", help="Corrupted audio directory")
    p_eval.add_argument("--skip-calibration", action="store_true", help="Skip threshold calibration")
    p_eval.set_defaults(func=_cmd_evaluate)

    # calibrate
    p_cal = subparsers.add_parser("calibrate", help="Calibrate synthetic score threshold")
    p_cal.add_argument("--natural-dir", default="data/clean", help="Natural audio directory")
    p_cal.add_argument("--synthetic-dir", default="data/synthetic", help="Synthetic audio directory")
    p_cal.set_defaults(func=_cmd_calibrate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
