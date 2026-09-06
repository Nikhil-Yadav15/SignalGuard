"""Benchmark audio synthesis presets matching dsp_project_prompt.md test suites."""

from __future__ import annotations

from io import BytesIO
import numpy as np
import soundfile as sf

PRESET_OPTIONS = [
    "None (Use uploaded file)",
    "Preset: Clean Speech Simulation (Natural prosody)",
    "Preset: Synthetic Speech Artifacts (Constant F0 + High Flatness)",
    "Preset: Powerline Hum (50 Hz + Harmonics)",
    "Preset: Additive Gaussian Noise (SNR ~10 dB)",
    "Preset: Clipped / Saturated Audio",
    "Preset: Impulse Click Spikes",
    "Preset: Dropout Glitches (Gaps)",
]

_PRESET_MAP = {
    "Preset: Clean Speech Simulation (Natural prosody)": "clean_harmonic",
    "Preset: Synthetic Speech Artifacts (Constant F0 + High Flatness)": "synthetic_speech",
    "Preset: Powerline Hum (50 Hz + Harmonics)": "powerline_hum",
    "Preset: Additive Gaussian Noise (SNR ~10 dB)": "gaussian_noise",
    "Preset: Clipped / Saturated Audio": "clipping",
    "Preset: Impulse Click Spikes": "impulse_clicks",
    "Preset: Dropout Glitches (Gaps)": "dropouts",
}


def get_preset_key(choice_name: str) -> str:
    """Resolve user selectbox choice to an internal generator key."""
    return _PRESET_MAP.get(choice_name, "clean_harmonic")


def generate_preset_audio(preset_key: str, sample_rate: int = 16_000) -> tuple[bytes, str]:
    """Synthesize deterministic benchmark audio matching dsp_project_prompt.md test classes."""
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)

    if preset_key == "clean_harmonic":
        # Natural speech simulation: time-varying F0 vibrato with harmonic comb
        f0_mod = 180.0 + 35.0 * np.sin(2 * np.pi * 1.5 * t) + 15.0 * np.sin(2 * np.pi * 3.7 * t)
        phase = 2 * np.pi * np.cumsum(f0_mod) / sample_rate
        harmonics = (
            0.40 * np.sin(phase)
            + 0.25 * np.sin(2 * phase)
            + 0.15 * np.sin(3 * phase)
            + 0.08 * np.sin(4 * phase)
            + 0.04 * np.sin(5 * phase)
        )
        samples = (harmonics * 0.7).astype(np.float32)
        filename = "preset_clean_human_simulation.wav"

    elif preset_key == "synthetic_speech":
        # Synthetic speech signature: unnaturally static F0, high spectral flatness, and comb artifacts
        static_f0 = 220.0
        phase = 2 * np.pi * static_f0 * t
        robot_tone = 0.35 * np.sin(phase) + 0.20 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
        rng = np.random.default_rng(42)
        vocoder_noise = 0.06 * rng.standard_normal(len(t), dtype=np.float32)
        samples = (robot_tone + vocoder_noise).astype(np.float32)
        filename = "preset_synthetic_tts_anomaly.wav"

    elif preset_key == "powerline_hum":
        # Clean tone corrupted by 50 Hz powerline hum and harmonics
        base = 0.30 * np.sin(2 * np.pi * 220.0 * t)
        hum_50 = 0.25 * np.sin(2 * np.pi * 50.0 * t)
        hum_100 = 0.10 * np.sin(2 * np.pi * 100.0 * t)
        samples = (base + hum_50 + hum_100).astype(np.float32)
        filename = "preset_corrupted_hum_50hz.wav"

    elif preset_key == "gaussian_noise":
        # Natural harmonic tone degraded by heavy additive broadband Gaussian noise (SNR ~10 dB)
        base = 0.35 * np.sin(2 * np.pi * 200.0 * t) + 0.2 * np.sin(2 * np.pi * 400.0 * t)
        rng = np.random.default_rng(101)
        noise = 0.12 * rng.standard_normal(len(t), dtype=np.float32)
        samples = (base + noise).astype(np.float32)
        filename = "preset_corrupted_gaussian_noise.wav"

    elif preset_key == "clipping":
        # Heavily clipped audio with flat top peaks
        raw = 1.8 * np.sin(2 * np.pi * 220.0 * t)
        samples = np.clip(raw, -0.98, 0.98).astype(np.float32)
        filename = "preset_corrupted_clipping.wav"

    elif preset_key == "impulse_clicks":
        # Tone corrupted by sharp non-sustained impulse clicks
        samples = 0.30 * np.sin(2 * np.pi * 240.0 * t).astype(np.float32)
        click_indices = [int(0.3 * sample_rate), int(0.9 * sample_rate), int(1.6 * sample_rate), int(2.4 * sample_rate)]
        for idx in click_indices:
            if idx < len(samples):
                samples[idx] = 1.0
                if idx + 1 < len(samples):
                    samples[idx + 1] = -0.95
        filename = "preset_corrupted_impulse_clicks.wav"

    elif preset_key == "dropouts":
        # Tone interrupted by contiguous silence gaps (glitches)
        samples = 0.35 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
        gap1_start, gap1_end = int(0.7 * sample_rate), int(0.74 * sample_rate)
        gap2_start, gap2_end = int(1.8 * sample_rate), int(1.83 * sample_rate)
        samples[gap1_start:gap1_end] = 0.0
        samples[gap2_start:gap2_end] = 0.0
        filename = "preset_corrupted_dropouts.wav"

    else:
        samples = (0.3 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
        filename = "preset_default_tone.wav"

    buf = BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue(), filename
