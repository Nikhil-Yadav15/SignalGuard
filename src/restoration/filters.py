"""Targeted deterministic DSP restoration primitives."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal

from ..detectors._common import (
    mono_audio,
    require_finite_real,
    require_positive_integer,
)


def spectral_subtract(
    samples: ArrayLike,
    sample_rate: int,
    settings: Mapping[str, object],
) -> NDArray[np.float32]:
    """Apply conservative spectral subtraction from low-energy frames."""

    x, rate = mono_audio(samples, sample_rate)
    if settings.get("method") != "spectral_subtraction":
        raise ValueError("noise-reduction method must be 'spectral_subtraction'")
    n_fft = require_positive_integer(settings["n_fft"], "noise_reduction.n_fft")
    hop = require_positive_integer(
        settings["hop_length"],
        "noise_reduction.hop_length",
    )
    if hop > n_fft:
        raise ValueError("noise_reduction.hop_length must not exceed n_fft")
    reference_quantile = require_finite_real(
        settings["noise_reference_quantile"],
        "noise_reduction.noise_reference_quantile",
        minimum=0.0,
    )
    if not 0.0 < reference_quantile <= 1.0:
        raise ValueError("noise_reference_quantile must be in (0, 1]")
    subtraction_factor = require_finite_real(
        settings["subtraction_factor"],
        "noise_reduction.subtraction_factor",
        minimum=0.0,
    )
    spectral_floor = require_finite_real(
        settings["spectral_floor_ratio"],
        "noise_reduction.spectral_floor_ratio",
        minimum=0.0,
    )
    if spectral_floor > 1.0:
        raise ValueError("spectral_floor_ratio must not exceed 1")
    if x.size < n_fft:
        return x.copy()

    _, _, stft = signal.stft(
        x.astype(np.float64),
        fs=rate,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        boundary="zeros",
        padded=True,
        window="hann",
    )
    frame_power = np.mean(np.abs(stft) ** 2, axis=0)
    reference_count = max(
        1,
        int(np.ceil(frame_power.size * reference_quantile)),
    )
    reference_indexes = np.argsort(frame_power)[:reference_count]
    noise_magnitude = np.mean(
        np.abs(stft[:, reference_indexes]),
        axis=1,
        keepdims=True,
    )
    magnitude = np.abs(stft)
    phase = np.exp(1j * np.angle(stft))
    restored_magnitude = np.maximum(
        magnitude - subtraction_factor * noise_magnitude,
        spectral_floor * noise_magnitude,
    )
    _, restored = signal.istft(
        restored_magnitude * phase,
        fs=rate,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        input_onesided=True,
        boundary=True,
        window="hann",
    )
    return _finish(restored, x.size)


def notch_hum(
    samples: ArrayLike,
    sample_rate: int,
    *,
    mains_hz: float,
    harmonic_count: int,
    quality_factor: float,
) -> NDArray[np.float32]:
    """Apply narrow IIR notches at mains frequency and its harmonics."""

    x, rate = mono_audio(samples, sample_rate)
    mains = require_finite_real(mains_hz, "mains_hz", minimum=0.0)
    if mains == 0.0:
        raise ValueError("mains_hz must be positive")
    harmonics = require_positive_integer(harmonic_count, "harmonic_count")
    quality = require_finite_real(
        quality_factor,
        "quality_factor",
        minimum=0.0,
    )
    if quality == 0.0:
        raise ValueError("quality_factor must be positive")

    restored = x.astype(np.float64)
    for harmonic in range(1, harmonics + 1):
        frequency = mains * harmonic
        if frequency >= rate / 2:
            break
        numerator, denominator = signal.iirnotch(
            frequency,
            quality,
            fs=rate,
        )
        if restored.size > 3 * max(len(numerator), len(denominator)):
            restored = signal.filtfilt(numerator, denominator, restored)
        else:
            restored = signal.lfilter(numerator, denominator, restored)
    return _finish(restored, x.size)


def repair_impulses(
    samples: ArrayLike,
    sample_rate: int,
    *,
    kernel_size: int,
    context_samples: int,
    threshold_multiplier: float = 8.0,
    minimum_absolute_amplitude: float = 0.20,
    minimum_residual_amplitude: float = 0.05,
) -> NDArray[np.float32]:
    """Detect local impulse outliers and interpolate only those samples."""

    x, _ = mono_audio(samples, sample_rate)
    kernel = require_positive_integer(kernel_size, "kernel_size")
    context = require_positive_integer(context_samples, "context_samples")
    if kernel < 3 or kernel % 2 == 0:
        raise ValueError("kernel_size must be odd and at least 3")
    multiplier = require_finite_real(
        threshold_multiplier,
        "threshold_multiplier",
        minimum=0.0,
    )
    minimum_amplitude = require_finite_real(
        minimum_absolute_amplitude,
        "minimum_absolute_amplitude",
        minimum=0.0,
    )
    minimum_residual = require_finite_real(
        minimum_residual_amplitude,
        "minimum_residual_amplitude",
        minimum=0.0,
    )

    half_width = kernel // 2
    padded = np.pad(x.astype(np.float64), half_width, mode="edge")
    neighborhoods = np.stack(
        [padded[offset : offset + x.size] for offset in range(kernel)]
    )
    local_median = np.median(neighborhoods, axis=0)
    local_mad = np.median(
        np.abs(neighborhoods - local_median),
        axis=0,
    )
    deviation = np.abs(x - local_median)
    impulse_mask = (
        deviation > multiplier * np.maximum(local_mad, 1e-6)
    ) & (deviation >= minimum_residual) & (np.abs(x) >= minimum_amplitude)
    restored = _interpolate_masked_runs(
        x.astype(np.float64),
        impulse_mask,
        context,
        fallback=local_median,
    )
    return _finish(restored, x.size)


def repair_dropouts(
    samples: ArrayLike,
    sample_rate: int,
    *,
    near_zero_amplitude: float,
    maximum_duration_ms: float,
    minimum_duration_ms: float = 0.0,
) -> NDArray[np.float32]:
    """Linearly interpolate bounded near-zero runs within duration limits."""

    x, rate = mono_audio(samples, sample_rate)
    amplitude = require_finite_real(
        near_zero_amplitude,
        "near_zero_amplitude",
        minimum=0.0,
    )
    maximum_ms = require_finite_real(
        maximum_duration_ms,
        "maximum_duration_ms",
        minimum=0.0,
    )
    minimum_ms = require_finite_real(
        minimum_duration_ms,
        "minimum_duration_ms",
        minimum=0.0,
    )
    if maximum_ms <= 0.0 or maximum_ms < minimum_ms:
        raise ValueError(
            "dropout duration limits must satisfy 0 <= minimum <= maximum"
        )
    minimum_count = max(1, int(round(minimum_ms * rate / 1000.0)))
    maximum_count = max(1, int(round(maximum_ms * rate / 1000.0)))
    candidate = np.abs(x) <= amplitude
    repair_mask = _run_length_filter(candidate, minimum_count, maximum_count)
    restored = _interpolate_masked_runs(
        x.astype(np.float64),
        repair_mask,
        context_samples=1,
        fallback=None,
    )
    return _finish(restored, x.size)


def highpass_rumble(
    samples: ArrayLike,
    sample_rate: int,
    *,
    cutoff_hz: float,
    order: int,
) -> NDArray[np.float32]:
    return _sos_filter(samples, sample_rate, cutoff_hz, order, "highpass")


def lowpass_hiss(
    samples: ArrayLike,
    sample_rate: int,
    *,
    cutoff_hz: float,
    order: int,
) -> NDArray[np.float32]:
    return _sos_filter(samples, sample_rate, cutoff_hz, order, "lowpass")


def repair_clipping(
    samples: ArrayLike,
    sample_rate: int,
    *,
    sample_level: float,
    context_samples: int,
    flat_top_min_samples: int = 1,
    flat_top_tolerance: float = 0.001,
) -> NDArray[np.float32]:
    """Interpolate qualified flat-top runs using surrounding context medians."""

    x, _ = mono_audio(samples, sample_rate)
    level = require_finite_real(sample_level, "sample_level", minimum=0.0)
    if level <= 0.0 or level > 1.0:
        raise ValueError("sample_level must be in (0, 1]")
    context = require_positive_integer(context_samples, "context_samples")
    minimum_run = require_positive_integer(
        flat_top_min_samples,
        "flat_top_min_samples",
    )
    tolerance = require_finite_real(
        flat_top_tolerance,
        "flat_top_tolerance",
        minimum=0.0,
    )
    clipped = _flat_top_run_filter(
        np.abs(x),
        level,
        tolerance,
        minimum_run,
    )
    restored = _interpolate_masked_runs(
        x.astype(np.float64),
        clipped,
        context,
        fallback=None,
    )
    return _finish(restored, x.size)


def _sos_filter(
    samples: ArrayLike,
    sample_rate: int,
    cutoff_hz: float,
    order: int,
    kind: str,
) -> NDArray[np.float32]:
    x, rate = mono_audio(samples, sample_rate)
    cutoff = require_finite_real(cutoff_hz, "cutoff_hz", minimum=0.0)
    filter_order = require_positive_integer(order, "order")
    if not 0.0 < cutoff < rate / 2:
        raise ValueError("cutoff_hz must be positive and below Nyquist")
    sections = signal.butter(
        filter_order,
        cutoff,
        btype=kind,
        fs=rate,
        output="sos",
    )
    padding_requirement = 3 * (2 * len(sections) + 1)
    if x.size > padding_requirement:
        restored = signal.sosfiltfilt(sections, x)
    else:
        restored = signal.sosfilt(sections, x)
    return _finish(restored, x.size)


def _run_length_filter(
    mask: np.ndarray,
    minimum_count: int,
    maximum_count: int,
) -> np.ndarray:
    result = np.zeros(mask.shape, dtype=bool)
    start: int | None = None
    for index, active in enumerate(np.r_[mask, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            length = index - start
            if minimum_count <= length <= maximum_count:
                result[start:index] = True
            start = None
    return result


def _flat_top_run_filter(
    absolute_samples: np.ndarray,
    sample_level: float,
    tolerance: float,
    minimum_count: int,
) -> np.ndarray:
    result = np.zeros(absolute_samples.shape, dtype=bool)
    candidate = absolute_samples >= sample_level
    start: int | None = None
    for index, active in enumerate(np.r_[candidate, False]):
        continues_flat = (
            active
            and start is not None
            and index > 0
            and abs(absolute_samples[index] - absolute_samples[index - 1])
            <= tolerance
        )
        if active and start is None:
            start = index
        elif active and not continues_flat:
            if start is not None and index - start >= minimum_count:
                result[start:index] = True
            start = index
        elif not active and start is not None:
            if index - start >= minimum_count:
                result[start:index] = True
            start = None
    return result


def _interpolate_masked_runs(
    values: np.ndarray,
    mask: np.ndarray,
    context_samples: int,
    *,
    fallback: np.ndarray | None,
) -> np.ndarray:
    result = values.copy()
    start: int | None = None
    for index, active in enumerate(np.r_[mask, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            end = index
            left_start = max(0, start - context_samples)
            right_end = min(values.size, end + context_samples)
            left_context = result[left_start:start]
            right_context = result[end:right_end]
            if left_context.size and right_context.size:
                left_value = float(np.median(left_context))
                right_value = float(np.median(right_context))
                result[start:end] = np.linspace(
                    left_value,
                    right_value,
                    end - start + 2,
                )[1:-1]
            elif fallback is not None:
                result[start:end] = fallback[start:end]
            start = None
    return result


def _finish(values: ArrayLike, length: int) -> NDArray[np.float32]:
    restored = np.asarray(values, dtype=np.float64)[:length]
    if restored.size < length:
        restored = np.pad(restored, (0, length - restored.size))
    if not np.all(np.isfinite(restored)):
        raise ValueError("restoration produced non-finite samples")
    return np.ascontiguousarray(np.clip(restored, -1.0, 1.0), dtype=np.float32)
