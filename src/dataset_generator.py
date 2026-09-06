"""Deterministic dataset generator for SignalGuard evaluation and calibration.

Synthesizes benchmark datasets into data/clean, data/synthetic, and data/corrupted
matching Section 4 (Data and Test Strategy) of dsp_project_prompt.md.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf


def generate_clean_speech(
    f0_base: float = 180.0,
    duration: float = 3.5,
    sample_rate: int = 16_000,
    seed: int = 0,
) -> np.ndarray:
    """Synthesize clean speech-like audio with natural pitch vibrato and harmonic decay."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)

    # Prosodic frequency modulation (vibrato + natural speech inflections)
    f0_mod = (
        f0_base
        + 25.0 * np.sin(2 * np.pi * 1.6 * t)
        + 12.0 * np.sin(2 * np.pi * 3.4 * t)
        + 6.0 * rng.standard_normal(len(t)) * 0.05
    )
    phase = 2 * np.pi * np.cumsum(f0_mod) / sample_rate

    # Natural harmonic series with 1/n rolloff
    signal = np.zeros_like(t)
    harmonic_amplitudes = [0.42, 0.26, 0.15, 0.08, 0.04, 0.02, 0.01]
    for h_idx, amp in enumerate(harmonic_amplitudes, start=1):
        signal += amp * np.sin(h_idx * phase)

    # Stationary high-fidelity recording
    samples = (signal * 0.70).astype(np.float32)
    return samples


def generate_synthetic_speech(
    f0_base: float = 210.0,
    duration: float = 3.5,
    sample_rate: int = 16_000,
    seed: int = 0,
) -> np.ndarray:
    """Synthesize speech with characteristic TTS / neural vocoder anomalies."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)

    # Anomaly 1: Overly static monotone F0 (no pitch variation)
    phase = 2 * np.pi * f0_base * t

    # Anomaly 2: Comb with unnatural equal amplitudes across harmonics
    tone = (
        0.30 * np.sin(phase)
        + 0.25 * np.sin(2 * phase)
        + 0.22 * np.sin(3 * phase)
        + 0.18 * np.sin(4 * phase)
        + 0.15 * np.sin(5 * phase)
    )

    # Anomaly 3: Inharmonic vocoder artifacts (fractional multiples)
    tone += 0.20 * np.sin(2 * np.pi * 2.5 * f0_base * t)
    tone += 0.18 * np.sin(2 * np.pi * 3.7 * f0_base * t)

    # Anomaly 4: High spectral flatness vocoder background noise
    vocoder_noise = 0.08 * rng.standard_normal(len(t), dtype=np.float32)

    samples = (tone + vocoder_noise).astype(np.float32)
    return samples * 0.6


def corrupt_additive_noise(samples: np.ndarray, target_snr_db: float = 12.0, seed: int = 42) -> np.ndarray:
    """Add broadband Gaussian noise to achieve target SNR."""
    rng = np.random.default_rng(seed)
    sig_power = float(np.mean(samples ** 2))
    noise_power = sig_power / (10.0 ** (target_snr_db / 10.0))
    noise = rng.normal(0, np.sqrt(noise_power), len(samples)).astype(np.float32)
    return samples + noise


def corrupt_hum(samples: np.ndarray, mains_freq: float = 50.0, sample_rate: int = 16_000) -> np.ndarray:
    """Add powerline mains hum and harmonics."""
    t = np.linspace(0, len(samples) / sample_rate, len(samples), endpoint=False, dtype=np.float32)
    hum = 0.25 * np.sin(2 * np.pi * mains_freq * t) + 0.10 * np.sin(2 * np.pi * 2 * mains_freq * t)
    return samples + hum


def corrupt_clipping(samples: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """Apply severe amplitude clipping saturation."""
    scaled = samples * (1.0 / threshold)
    return np.clip(scaled, -0.98, 0.98).astype(np.float32)


def corrupt_impulses(samples: np.ndarray, count: int = 8, seed: int = 42) -> np.ndarray:
    """Inject isolated non-sustained click impulse spikes."""
    rng = np.random.default_rng(seed)
    corrupted = samples.copy()
    indices = rng.choice(len(samples) - 2, size=count, replace=False)
    for idx in indices:
        corrupted[idx] = rng.choice([-0.95, 0.95])
    return corrupted


def corrupt_dropouts(samples: np.ndarray, sample_rate: int = 16_000, gap_ms: float = 40.0) -> np.ndarray:
    """Zero out contiguous audio segments to simulate transmission dropouts."""
    corrupted = samples.copy()
    gap_samples = int(sample_rate * gap_ms / 1000.0)
    for point in [int(0.8 * sample_rate), int(1.8 * sample_rate), int(2.6 * sample_rate)]:
        if point + gap_samples < len(corrupted):
            corrupted[point : point + gap_samples] = 0.0
    return corrupted


def generate_benchmark_dataset(
    output_dir: str | Path = "data",
    samples_per_class: int = 10,
    sample_rate: int = 16_000,
) -> dict[str, int]:
    """Generate balanced benchmark datasets matching Section 4 of dsp_project_prompt.md."""
    base_dir = Path(output_dir).resolve()
    clean_dir = base_dir / "clean"
    synthetic_dir = base_dir / "synthetic"
    corrupted_dir = base_dir / "corrupted"

    for d in (clean_dir, synthetic_dir, corrupted_dir):
        d.mkdir(parents=True, exist_ok=True)

    counts = {"clean": 0, "synthetic": 0, "corrupted": 0}

    # 1. Clean Human Speech Simulations
    pitches = [120.0, 150.0, 175.0, 200.0, 225.0, 250.0, 140.0, 190.0, 210.0, 260.0]
    for i in range(samples_per_class):
        f0 = pitches[i % len(pitches)]
        audio = generate_clean_speech(f0_base=f0, duration=3.5, sample_rate=sample_rate, seed=100 + i)
        out_path = clean_dir / f"clean_speech_{i+1:03d}_{int(f0)}hz.wav"
        sf.write(out_path, audio, sample_rate, format="WAV", subtype="PCM_16")
        counts["clean"] += 1

    # 2. Synthetic Audio Anomalies
    for i in range(samples_per_class):
        f0 = pitches[i % len(pitches)]
        audio = generate_synthetic_speech(f0_base=f0, duration=3.5, sample_rate=sample_rate, seed=200 + i)
        out_path = synthetic_dir / f"synthetic_tts_{i+1:03d}_{int(f0)}hz.wav"
        sf.write(out_path, audio, sample_rate, format="WAV", subtype="PCM_16")
        counts["synthetic"] += 1

    # 3. Corrupted Human Audio Variations
    corruption_types = ["noise", "hum", "clipping", "impulse", "dropout"]
    for i in range(samples_per_class):
        f0 = pitches[i % len(pitches)]
        clean_base = generate_clean_speech(f0_base=f0, duration=3.5, sample_rate=sample_rate, seed=300 + i)
        ctype = corruption_types[i % len(corruption_types)]

        if ctype == "noise":
            audio = corrupt_additive_noise(clean_base, target_snr_db=12.0, seed=400 + i)
        elif ctype == "hum":
            audio = corrupt_hum(clean_base, mains_freq=50.0 if i % 2 == 0 else 60.0, sample_rate=sample_rate)
        elif ctype == "clipping":
            audio = corrupt_clipping(clean_base, threshold=0.5)
        elif ctype == "impulse":
            audio = corrupt_impulses(clean_base, count=8, seed=500 + i)
        else:
            audio = corrupt_dropouts(clean_base, sample_rate=sample_rate, gap_ms=40.0)

        out_path = corrupted_dir / f"corrupted_{ctype}_{i+1:03d}.wav"
        sf.write(out_path, audio, sample_rate, format="WAV", subtype="PCM_16")
        counts["corrupted"] += 1

    return counts


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate SignalGuard benchmark audio dataset")
    parser.add_argument("--output-dir", default="data", help="Target output directory (default: data)")
    parser.add_argument("--count", type=int, default=10, help="Number of files per class (default: 10)")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Target sample rate (default: 16000)")
    args = parser.parse_args(argv)

    print(f"Synthesizing benchmark dataset ({args.count} files per class) into '{args.output_dir}'...")
    counts = generate_benchmark_dataset(
        output_dir=args.output_dir,
        samples_per_class=args.count,
        sample_rate=args.sample_rate,
    )
    print("Dataset generation completed:")
    for k, v in counts.items():
        print(f"  • data/{k}: {v} WAV files")


if __name__ == "__main__":
    main()
