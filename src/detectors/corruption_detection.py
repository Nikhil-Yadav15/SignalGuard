"""Rule-based corruption detection for targeted, reversible DSP repair."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from ._common import frames, mono_audio
from ..utils import get_config_section, load_config, to_json_safe


class Severity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


@dataclass(frozen=True, slots=True)
class DetectedCorruption:
    kind: str
    detected: bool
    severity: Severity | None
    metric_name: str
    metric_value: float | None
    threshold: float
    explanation: str


@dataclass(frozen=True, slots=True)
class CorruptionReport:
    sample_rate_hz: int
    detections: tuple[DetectedCorruption, ...]
    metrics: dict[str, float | int | None]
    warnings: tuple[str, ...] = ()
    def to_dict(self) -> dict[str, Any]: return to_json_safe(self)
    @property
    def detected_kinds(self) -> tuple[str, ...]: return tuple(item.kind for item in self.detections if item.detected)


def detect_corruption(samples: ArrayLike, sample_rate: int, *, config: Mapping[str, Any] | None = None) -> CorruptionReport:
    values, rate = mono_audio(samples, sample_rate)
    root = load_config() if config is None else config
    section = get_config_section(root, "corruption_detection")
    noise = _noise_metric(values, section["broadband_noise"])
    clipping, flat_top_runs = _clipping_metric(values, section["clipping"])
    hum = _hum_metric(values, rate, section["hum"])
    impulses = _impulse_metric(values, rate, section["impulse"])
    dropouts = _dropout_metric(values, rate, section["dropout"])
    rumble = _band_ratio(values, rate, upper=float(section["rumble"]["cutoff_hz"]))
    hiss = _band_ratio(values, rate, lower=float(section["hiss"]["lower_frequency_hz"]))
    det = (
        _lower("broadband_noise", "estimated_snr_db", noise, float(section["broadband_noise"]["detection_snr_db"]), section["broadband_noise"]["severity_boundaries_snr_db"]),
        _clipping_detection(
            clipping,
            float(section["clipping"]["mild_ratio"]),
            section["clipping"],
        ),
        _higher("hum", "peak_to_local_ratio_db", hum, float(section["hum"]["minimum_peak_to_local_ratio_db"]), section["hum"]["severity_boundaries_prominence_db"]),
        _higher("impulse", "events_per_second", impulses, float(section["impulse"]["detection_events_per_second"]), section["impulse"]["severity_boundaries_events_per_second"]),
        _ratio("dropout", "total_dropout_ratio", dropouts[0], float(section["dropout"]["detection_total_ratio"]), section["dropout"]["severity_boundaries_total_ratio"], ("moderate_above", "high_above")),
        _ratio("rumble", "low_band_energy_ratio", rumble, float(section["rumble"]["excessive_energy_ratio"]), section["rumble"]["severity_boundaries_energy_ratio"], ("moderate_above", "high_above")),
        _ratio("hiss", "high_band_energy_ratio", hiss, float(section["hiss"]["excessive_energy_ratio"]), section["hiss"]["severity_boundaries_energy_ratio"], ("moderate_above", "high_above")),
    )
    metrics = {
        "estimated_snr_db": noise,
        "clipped_sample_ratio": clipping,
        "flat_top_run_count": flat_top_runs,
        "hum_prominence_db": hum,
        "impulse_events_per_second": impulses,
        "total_dropout_ratio": dropouts[0],
        "dropout_count": dropouts[1],
        "low_band_energy_ratio": rumble,
        "high_band_energy_ratio": hiss,
    }
    warnings = (
        ("SNR was not estimable because frame energy lacked a usable noise reference.",)
        if noise is None
        else ()
    )
    return CorruptionReport(rate, det, metrics, warnings)


def _noise_metric(x: np.ndarray, settings: Mapping[str, Any]) -> float | None:
    block = frames(x, int(settings["frame_length"]), int(settings["hop_length"]))
    if not len(block): return None
    power = np.mean(block ** 2, axis=1)
    # A perfectly stationary recording has no defensible low-energy noise reference.
    if np.ptp(10 * np.log10(np.maximum(power, np.finfo(float).tiny))) < 1.0: return None
    floor = float(np.mean(np.sort(power)[:max(1, int(np.ceil(len(power) * float(settings["low_energy_quantile"]))))]))
    high = float(np.mean(np.sort(power)[-max(1, int(np.ceil(len(power) * float(settings["low_energy_quantile"])))):]))
    return float(10 * np.log10(max(high - floor, np.finfo(float).tiny) / max(floor, np.finfo(float).tiny)))


def _clipping_metric(
    x: np.ndarray,
    settings: Mapping[str, Any],
) -> tuple[float, int]:
    qualified, run_count = _flat_top_mask(
        np.abs(x),
        float(settings["sample_level"]),
        float(settings["flat_top_tolerance"]),
        int(settings["flat_top_min_samples"]),
    )
    return float(np.mean(qualified)), run_count


def _hum_metric(x: np.ndarray, rate: int, settings: Mapping[str, Any]) -> float | None:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freq = np.fft.rfftfreq(x.size, 1 / rate)
    local = float(settings["local_bandwidth_hz"])
    tol = float(settings["search_tolerance_hz"])
    harmonic_values: dict[int, float] = {}
    for harmonic in range(1, int(settings["harmonic_count"]) + 1):
        target = float(settings["mains_frequency_hz"]) * harmonic
        if target >= rate / 2:
            break
        peak_mask = np.abs(freq - target) <= tol
        nearby = (np.abs(freq - target) <= local) & ~peak_mask
        # Short clips can have FFT bins wider than either configured window.
        # Such clips do not contain enough resolution for defensible hum evidence.
        if not np.any(peak_mask) or not np.any(nearby):
            continue
        peak = float(np.mean(spectrum[peak_mask]))
        baseline = float(np.mean(spectrum[nearby]))
        harmonic_values[harmonic] = 10 * np.log10(
            max(peak, np.finfo(float).tiny) / max(baseline, np.finfo(float).tiny)
        )
    thresh = float(settings["minimum_peak_to_local_ratio_db"])
    strong_harmonics = [h for h, val in harmonic_values.items() if val >= thresh]
    # Mains hum physically requires the fundamental mains frequency (h=1) to be elevated,
    # plus at least one supporting harmonic (h >= 2) to rule out single tonal components.
    # Normal voice pitch harmonics at 100 Hz, 150 Hz, etc. do not have 50 Hz mains fundamental.
    if 1 in strong_harmonics and len(strong_harmonics) >= 2:
        return float(np.mean([harmonic_values[h] for h in strong_harmonics]))
    return None


def _impulse_metric(x: np.ndarray, rate: int, settings: Mapping[str, Any]) -> float:
    width=int(settings["local_window_samples"]); pad=width//2; padded=np.pad(x.astype(float), pad, mode="edge")
    local=np.stack([padded[i:i+x.size] for i in range(width)])
    median=np.median(local,axis=0); mad=np.median(np.abs(local-median),axis=0)
    deviation = np.abs(x - median)
    flagged=(deviation>float(settings["median_absolute_deviation_multiplier"])*np.maximum(mad,1e-6))&(deviation>=float(settings["minimum_residual_amplitude"]))&(np.abs(x)>=float(settings["minimum_absolute_amplitude"]))
    return float(_count_true_runs(flagged) / (x.size / rate))


def _dropout_metric(x: np.ndarray, rate: int, settings: Mapping[str, Any]) -> tuple[float, int]:
    mask=np.abs(x)<=float(settings["near_zero_amplitude"]); threshold=int(round(float(settings["minimum_duration_ms"])*rate/1000))
    count=0; covered=0; start=None
    for index, active in enumerate(np.r_[mask, False]):
        if active and start is None: start=index
        elif not active and start is not None:
            length=index-start
            if length>=threshold: count+=1; covered+=length
            start=None
    return float(covered/x.size), count


def _band_ratio(x: np.ndarray, rate: int, *, lower: float | None = None, upper: float | None = None) -> float:
    power=np.abs(np.fft.rfft(x*np.hanning(x.size)))**2; freq=np.fft.rfftfreq(x.size,1/rate); mask=np.ones(freq.size,bool)
    if lower is not None: mask &= freq>=lower
    if upper is not None: mask &= freq<=upper
    mask[0] = False
    denominator = float(np.sum(power[1:]))
    if denominator <= np.finfo(float).tiny:
        return 0.0
    return float(np.clip(np.sum(power[mask]) / denominator, 0.0, 1.0))


def _severity(value: float, boundaries: Mapping[str, Any], *, lower_is_worse: bool = False) -> Severity:
    moderate=float(boundaries["moderate_below" if lower_is_worse else "moderate_above"]); high=float(boundaries["high_below" if lower_is_worse else "high_above"])
    if (value <= high if lower_is_worse else value >= high): return Severity.HIGH
    if (value <= moderate if lower_is_worse else value >= moderate): return Severity.MODERATE
    return Severity.LOW
def _lower(kind,name,value,threshold,boundaries): return DetectedCorruption(kind,value is not None and value<threshold,_severity(value,boundaries,lower_is_worse=True) if value is not None and value<threshold else None,name,value,threshold,f"{name} is below the configured threshold.")
def _higher(kind,name,value,threshold,boundaries): return DetectedCorruption(kind,value is not None and value>threshold,_severity(value,boundaries) if value is not None and value>threshold else None,name,value,threshold,f"{name} exceeds the configured threshold.")
def _ratio(kind,name,value,threshold,boundaries,keys):
    if "moderate_ratio" in keys: b={"moderate_above":boundaries[keys[0]],"high_above":boundaries[keys[1]]}
    else: b=boundaries
    return _higher(kind,name,value,threshold,b)


def _clipping_detection(
    value: float,
    threshold: float,
    settings: Mapping[str, Any],
) -> DetectedCorruption:
    detected = value > threshold
    severity: Severity | None = None
    if detected:
        if value >= float(settings["severe_ratio"]):
            severity = Severity.SEVERE
        elif value >= float(settings["moderate_ratio"]):
            severity = Severity.MODERATE
        else:
            severity = Severity.LOW
    return DetectedCorruption(
        "clipping",
        detected,
        severity,
        "clipped_sample_ratio",
        value,
        threshold,
        "Qualified flat-top clipping ratio is compared with the configured threshold.",
    )


def _flat_top_mask(
    absolute_samples: np.ndarray,
    sample_level: float,
    tolerance: float,
    minimum_run_length: int,
) -> tuple[np.ndarray, int]:
    """Return near-constant high-amplitude runs, excluding smooth sine peaks."""

    if tolerance < 0:
        raise ValueError("flat-top tolerance must be non-negative")
    if minimum_run_length <= 0:
        raise ValueError("minimum_run_length must be positive")
    candidate = absolute_samples >= sample_level
    qualified = np.zeros(candidate.shape, dtype=bool)
    run_count = 0
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
            if start is not None and index - start >= minimum_run_length:
                qualified[start:index] = True
                run_count += 1
            start = index
        elif not active and start is not None:
            if index - start >= minimum_run_length:
                qualified[start:index] = True
                run_count += 1
            start = None
    return qualified, run_count


def _count_true_runs(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    return int(mask[0]) + int(np.count_nonzero(mask[1:] & ~mask[:-1]))
