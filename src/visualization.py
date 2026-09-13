"""Non-interactive Matplotlib views for SignalGuard results."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from tempfile import gettempdir

with contextlib.suppress(ImportError):
    import numba  # noqa: F401 - eagerly initialize to prevent lazy_loader circular imports

# Some locked-down desktop installations do not permit Matplotlib to create a
# cache below the user profile. Keep the cache in the platform temp directory.
_MPL_CACHE = Path(gettempdir()) / "signalguard-matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import librosa
import librosa.display
import matplotlib

# SDK visualization functions must work in CI and headless batch jobs as well
# as in Streamlit. Streamlit renders the returned Figure itself.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from src.detectors._common import mono_audio


def plot_waveform(
    samples: ArrayLike,
    sample_rate: int,
    *,
    title: str = "Waveform",
) -> Figure:
    x, rate = mono_audio(samples, sample_rate)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(np.arange(x.size) / rate, x, lw=0.7, color="#3b82f6")
    ax.set(title=title, xlabel="Time (s)", ylabel="Amplitude")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    return fig


def plot_spectrogram(
    samples: ArrayLike,
    sample_rate: int,
    *,
    title: str = "Spectrogram",
) -> Figure:
    x, rate = mono_audio(samples, sample_rate)
    fig, ax = plt.subplots(figsize=(10, 4))
    if x.size < 2:
        ax.text(
            0.5,
            0.5,
            "Insufficient audio for a spectrogram",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        return fig
    n_fft = min(1024, int(x.size))
    hop = max(1, min(256, n_fft // 4))
    spectrum = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop, center=False))
    image = librosa.amplitude_to_db(
        np.maximum(spectrum, np.finfo(float).tiny),
        ref=np.max,
    )
    display = librosa.display.specshow(
        image,
        sr=rate,
        hop_length=hop,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
    )
    ax.set_title(title)
    fig.colorbar(display, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def plot_before_after(
    before: ArrayLike,
    after: ArrayLike,
    sample_rate: int,
) -> Figure:
    b, rate = mono_audio(before, sample_rate)
    a, _ = mono_audio(after, rate)
    diff = b - a
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    time = np.arange(b.size) / rate

    axes[0].plot(time, b, lw=0.6, color="#ef4444")
    axes[0].set_title("1. Before Restoration (Degraded Input Audio)")
    axes[0].grid(alpha=0.25, linestyle="--")

    axes[1].plot(np.arange(a.size) / rate, a, lw=0.6, color="#10b981")
    axes[1].set_title("2. After Restoration (Repaired Speech)")
    axes[1].grid(alpha=0.25, linestyle="--")

    axes[2].plot(time, diff, lw=0.6, color="#f59e0b")
    axes[2].set_title("3. Extracted Noise / Removed Distortion (Difference: Before − After)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.25, linestyle="--")

    for axis in axes:
        axis.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def plot_before_after_spectrogram(
    before: ArrayLike,
    after: ArrayLike,
    sample_rate: int,
    *,
    title_before: str = "Before Restoration",
    title_after: str = "After Restoration",
) -> Figure:
    """Side-by-side spectrogram comparison of degraded vs restored audio, plus removed spectral energy."""
    b, rate = mono_audio(before, sample_rate)
    a, _ = mono_audio(after, rate)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)

    n_fft = min(1024, int(max(b.size, a.size)))
    hop = max(1, min(256, n_fft // 4))

    spec_b = np.abs(librosa.stft(b, n_fft=n_fft, hop_length=hop, center=False))
    spec_a = np.abs(librosa.stft(a, n_fft=n_fft, hop_length=hop, center=False))

    global_max = max(float(np.max(spec_b)), float(np.max(spec_a)), 1e-9)

    img_b = librosa.amplitude_to_db(np.maximum(spec_b, np.finfo(float).tiny), ref=global_max)
    img_a = librosa.amplitude_to_db(np.maximum(spec_a, np.finfo(float).tiny), ref=global_max)

    # Compute positive spectral attenuation (energy removed by filters)
    img_diff = np.maximum(0.0, img_b - img_a)
    max_diff_val = max(6.0, float(np.max(img_diff)))

    items = [
        (img_b, f"1. {title_before}", axes[0], "magma", -80.0, 0.0, "%+2.0f dB"),
        (img_a, f"2. {title_after}", axes[1], "magma", -80.0, 0.0, "%+2.0f dB"),
        (img_diff, "3. Removed Spectral Energy (dB)", axes[2], "viridis", 0.0, max_diff_val, "%.1f dB"),
    ]

    for img, title, ax, cmap, vmin, vmax, fmt in items:
        display = librosa.display.specshow(
            img,
            sr=rate,
            hop_length=hop,
            x_axis="time",
            y_axis="hz",
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        fig.colorbar(display, ax=ax, format=fmt)

    fig.tight_layout()
    return fig
