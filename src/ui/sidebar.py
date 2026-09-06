"""Sidebar documentation and reference tables for SignalGuard UI."""

from __future__ import annotations

import streamlit as st


def render_sidebar() -> None:
    """Render the sidebar documentation, architecture overview, and benchmark tables."""
    with st.sidebar:
        st.markdown("### 🛡️ SignalGuard")
        st.markdown(
            "**Deterministic Audio Quality & Forensics SDK**\n\n"
            "Pre-training quality-control layer for ML speech pipelines. "
            "Detects synthetic audio anomalies and applies targeted DSP restoration "
            "with **pure classical signal processing** (no ML classifiers)."
        )

        st.markdown("---")
        st.markdown("### 🔄 Pipeline Decision Logic")
        st.markdown(
            """
            ```text
            Input Audio
               │
               ▼
            Canonical Preprocess (Mono, 16 kHz)
               │
               ▼
            Segment (3.0s Windows)
               │
               ▼
            Forensic Tests (F0, Harmonics, Spec, Phase)
               │
             [Score > 70%?] ──►   REJECT (Likely Synthetic)
               │ (No)
               ▼
            Corruption Detection (Noise, Hum, Impulses...)
               │
             [Clean?] ────────►   PASS (Clean Speech)
               │ (Corrupt)
               ▼
            Targeted DSP Filters (Wiener, Notch, Median...)
               │
               ▼
            Post-Restoration Quality Gate (SNR Gain)
               │
             [Gain > 0 dB?] ──►   PASS (Restored)
               │ (No gain)
               ▼
              REJECT (Unrecoverable)
            ```
            """
        )

        st.markdown("---")
        st.markdown("### 📊 Forensic Benchmark Table")
        st.dataframe(
            [
                {"Metric": "F0 STD", "Human Norm": "10–50 Hz", "Synthetic Indicator": "<10 Hz (Too flat)"},
                {"Metric": "Zero-Crossing Rate", "Human Norm": "0.01–0.05", "Synthetic Indicator": ">0.08 (Noisy)"},
                {"Metric": "Spectral Flatness", "Human Norm": "0.10–0.30", "Synthetic Indicator": ">0.50 (White-noise like)"},
                {"Metric": "Spectral Centroid", "Human Norm": "Dynamic CV", "Synthetic Indicator": "Low CV (Monotone timbre)"},
                {"Metric": "Phase Continuity", "Human Norm": "Smooth drift", "Synthetic Indicator": "Jagged jumps or rigid linear"},
                {"Metric": "LPC Glottal Kurtosis", "Human Norm": ">5.0 (Impulsive)", "Synthetic Indicator": "<3.0 (Smeared/Gaussian)"},
                {"Metric": "Mean Bicoherence", "Human Norm": ">0.15 (Coupled QPC)", "Synthetic Indicator": "<0.08 (Decoupled harmonics)"},
                {"Metric": "Modulation Syllabic Ratio", "Human Norm": "Dynamic 2–8 Hz", "Synthetic Indicator": "Sharp peak (Robotic cadence)"},
                {"Metric": "Speech Respiration Pauses", "Human Norm": "Breaths every 2–4s", "Synthetic Indicator": "Run-on (>4.5s) or Dead 0s"},
                {"Metric": "Offset Reverberation Decay", "Human Norm": "Exponential 15–60 ms", "Synthetic Indicator": "Abrupt cutoff (<8 ms)"},
                {"Metric": "SNR", "Human Norm": ">25 dB (Clean)", "Synthetic Indicator": "<15 dB (Corrupted)"},
            ],
            hide_index=True,
            width="stretch",
        )

        st.markdown("---")
        st.markdown("### 🛠️ DSP Restoration Reference")
        st.dataframe(
            [
                {"Distortion": "Broadband Noise", "DSP Filter": "Wiener / Spectral Subtraction"},
                {"Distortion": "50/60 Hz Hum", "DSP Filter": "Digital IIR Notch (Q=30)"},
                {"Distortion": "Impulse / Clicks", "DSP Filter": "Median Filter (scipy)"},
                {"Distortion": "Dropouts / Glitches", "DSP Filter": "Linear / Spline Interpolation"},
                {"Distortion": "High-Freq Hiss", "DSP Filter": "Hamming Sinc LPF (8 kHz)"},
                {"Distortion": "Low-Freq Rumble", "DSP Filter": "Butterworth HPF (80 Hz)"},
            ],
            hide_index=True,
            width="stretch",
        )
