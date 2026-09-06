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
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    time = np.arange(b.size) / rate
    axes[0].plot(time, b, lw=0.6, color="#ef4444")
    axes[0].set_title("Before restoration")
    axes[0].grid(alpha=0.25, linestyle="--")
    axes[1].plot(np.arange(a.size) / rate, a, lw=0.6, color="#10b981")
    axes[1].set_title("After restoration")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.25, linestyle="--")
    for axis in axes:
        axis.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def plot_before_after_spectrogram(
    before: ArrayLike,
    after: ArrayLike,
    sample_rate: int,
    *,
    title_before: str = "Spectrogram (Before Restoration)",
    title_after: str = "Spectrogram (After Restoration)",
) -> Figure:
    """Side-by-side spectrogram comparison of degraded vs restored audio."""
    b, rate = mono_audio(before, sample_rate)
    a, _ = mono_audio(after, rate)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    items = [(b, title_before, axes[0]), (a, title_after, axes[1])]
    for _idx, (sig, title, ax) in enumerate(items):
        if sig.size < 2:
            ax.text(
                0.5,
                0.5,
                "Insufficient audio",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue
        n_fft = min(1024, int(sig.size))
        hop = max(1, min(256, n_fft // 4))
        spectrum = np.abs(librosa.stft(sig, n_fft=n_fft, hop_length=hop, center=False))
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
