"""Automated evaluation and benchmark suite matching Section 9 of dsp_project_prompt.md.

Produces end-to-end confusion matrices, forensic separation metrics,
restoration SNR gain reports, and calibration recommendations.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

with contextlib.suppress(ImportError):
    import numba  # noqa: F401

from src.calibration import calibrate_likely_synthetic_threshold, score_labelled_audio
from src.pipeline import PipelineDecision, SignalGuardPipeline


@dataclass(frozen=True)
class ClassEvaluation:
    condition: str
    sample_count: int
    correct_pass: int
    correct_reject: int
    false_pass: int
    false_reject: int
    avg_score: float
    snr_gains: tuple[float, ...]

    @property
    def accuracy(self) -> float:
        total = self.sample_count
        if total == 0:
            return 0.0
        return (self.correct_pass + self.correct_reject) / total


def evaluate_pipeline_on_datasets(
    clean_dir: str | Path = "data/clean",
    synthetic_dir: str | Path = "data/synthetic",
    corrupted_dir: str | Path = "data/corrupted",
) -> dict[str, ClassEvaluation]:
    """Run SignalGuard on all three dataset classes and compile Section 9 metrics."""
    pipeline = SignalGuardPipeline()
    results: dict[str, ClassEvaluation] = {}

    # 1. Clean Human Speech
    clean_path = Path(clean_dir)
    clean_files = sorted(clean_path.glob("*.wav"))
    clean_scores: list[float] = []
    c_pass = c_rej = f_pass = f_rej = 0
    for f in clean_files:
        res = pipeline.analyze_file(str(f))
        clean_scores.append(res.evidence.score)
        if res.decision is PipelineDecision.PASS:
            c_pass += 1
        else:
            f_rej += 1
    results["Clean Human"] = ClassEvaluation(
        condition="Clean Human",
        sample_count=len(clean_files),
        correct_pass=c_pass,
        correct_reject=c_rej,
        false_pass=f_pass,
        false_reject=f_rej,
        avg_score=float(np.mean(clean_scores)) if clean_scores else 0.0,
        snr_gains=(),
    )

    # 2. Synthetic Speech
    synth_path = Path(synthetic_dir)
    synth_files = sorted(synth_path.glob("*.wav"))
    synth_scores: list[float] = []
    c_pass = c_rej = f_pass = f_rej = 0
    for f in synth_files:
        res = pipeline.analyze_file(str(f))
        synth_scores.append(res.evidence.score)
        if res.decision in (
            PipelineDecision.REJECT_LIKELY_SYNTHETIC,
            PipelineDecision.REJECT_UNRECOVERABLE,
        ):
            c_rej += 1
        else:
            f_pass += 1
    results["AI-Generated"] = ClassEvaluation(
        condition="AI-Generated",
        sample_count=len(synth_files),
        correct_pass=c_pass,
        correct_reject=c_rej,
        false_pass=f_pass,
        false_reject=f_rej,
        avg_score=float(np.mean(synth_scores)) if synth_scores else 0.0,
        snr_gains=(),
    )

    # 3. Corrupted Human Speech
    corr_path = Path(corrupted_dir)
    corr_files = sorted(corr_path.glob("*.wav"))
    corr_scores: list[float] = []
    snr_gains: list[float] = []
    c_pass = c_rej = f_pass = f_rej = 0
    for f in corr_files:
        res = pipeline.analyze_file(str(f))
        corr_scores.append(res.evidence.score)
        if res.decision is PipelineDecision.PASS_RESTORED:
            c_pass += 1
            if res.quality:
                for chk in res.quality.to_dict()["checks"]:
                    if chk["code"] == "broadband_noise_reduction":
                        delta = chk.get("after_value", 0) - chk.get("before_value", 0)
                        snr_gains.append(delta)
        elif res.decision is PipelineDecision.REJECT_UNRECOVERABLE:
            c_rej += 1
        elif res.decision is PipelineDecision.PASS:
            f_pass += 1
        else:
            f_rej += 1

    results["Corrupted Human"] = ClassEvaluation(
        condition="Corrupted Human",
        sample_count=len(corr_files),
        correct_pass=c_pass,
        correct_reject=c_rej,
        false_pass=f_pass,
        false_reject=f_rej,
        avg_score=float(np.mean(corr_scores)) if corr_scores else 0.0,
        snr_gains=tuple(snr_gains),
    )

    return results


def print_evaluation_report(results: dict[str, ClassEvaluation]) -> None:
    """Format and print Section 9 evaluation report."""
    print("=" * 80)
    print(" SIGNALGUARD · COMPREHENSIVE PIPELINE EVALUATION REPORT (SECTION 9)")
    print("=" * 80)

    print("\n1. End-to-End Decision Matrix:")
    print("-" * 80)
    header = (
        f"{'Condition':<18} | {'N':<6} | {'PASS':<8} | {'REJECT':<8}"
        f" | {'False PASS':<12} | {'False REJ':<10} | {'Accuracy':<8}"
    )
    print(header)
    print("-" * 80)
    for res in results.values():
        row = (
            f"{res.condition:<18} | {res.sample_count:<6} | {res.correct_pass:<8} | "
            f"{res.correct_reject:<8} | {res.false_pass:<12} | {res.false_reject:<10} | "
            f"{res.accuracy * 100:6.1f}%"
        )
        print(row)
    print("-" * 80)

    print("\n2. Forensic Separation & Score Summary:")
    print("-" * 80)
    for name, res in results.items():
        print(f"  • {name:<16}: Mean Evidence Score = {res.avg_score:.2f}/100")

    corrupted = results.get("Corrupted Human")
    if corrupted and corrupted.snr_gains:
        avg_gain = float(np.mean(corrupted.snr_gains))
        print("\n3. Restoration SNR Improvement:")
        print(
            f"  • Mean SNR Improvement on Restored Noise/Corruptions: +{avg_gain:.2f} dB"
        )


def run_calibration(
    clean_dir: str | Path = "data/clean",
    synthetic_dir: str | Path = "data/synthetic",
) -> None:
    """Run threshold calibration across labelled clean and synthetic directories."""
    print("\n" + "=" * 80)
    print(" THRESHOLD CALIBRATION ANALYSIS")
    print("=" * 80)
    try:
        examples = score_labelled_audio(clean_dir, synthetic_dir)
        report = calibrate_likely_synthetic_threshold(examples)
        m = report.metrics
        print(
            f"  • Processed Samples: {report.example_count}"
            f" ({report.natural_count} natural, {report.synthetic_count} synthetic)"
        )
        print(
            f"  • Recommended Likely Synthetic Threshold : "
            f"{report.recommended_likely_synthetic_threshold:.1f}%"
        )
        print(f"  • Resulting F1 Score                     : {m.f1:.4f}")
        print(
            f"  • Precision / Recall / Specificity       : "
            f"{m.precision:.4f} / {m.recall:.4f} / {m.specificity:.4f}"
        )
        print(
            f"  • Confusion (TP / FP / TN / FN)          : "
            f"{m.true_positive} / {m.false_positive} / {m.true_negative} / {m.false_negative}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  Calibration error: {exc}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SignalGuard on benchmark datasets"
    )
    parser.add_argument(
        "--clean-dir", default="data/clean", help="Directory with clean audio"
    )
    parser.add_argument(
        "--synthetic-dir",
        default="data/synthetic",
        help="Directory with synthetic audio",
    )
    parser.add_argument(
        "--corrupted-dir",
        default="data/corrupted",
        help="Directory with corrupted audio",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip threshold calibration analysis",
    )
    args = parser.parse_args(argv)

    results = evaluate_pipeline_on_datasets(
        clean_dir=args.clean_dir,
        synthetic_dir=args.synthetic_dir,
        corrupted_dir=args.corrupted_dir,
    )
    print_evaluation_report(results)

    if not args.skip_calibration:
        run_calibration(clean_dir=args.clean_dir, synthetic_dir=args.synthetic_dir)


if __name__ == "__main__":
    main()
