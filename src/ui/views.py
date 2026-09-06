from __future__ import annotations

from io import BytesIO

import numpy as np
import soundfile as sf
import streamlit as st

from src.detectors._common import AnalysisResult
from src.detectors.f0_analysis import F0AnalysisResult
from src.pipeline import PipelineDecision, PipelineResult
from src.scoring import EvidenceDecision
from src.visualization import (
    plot_before_after,
    plot_before_after_spectrogram,
    plot_spectrogram,
    plot_waveform,
)


def render_header() -> None:
    """Render the top branding and badges."""
    st.markdown(
        """
        <div class="main-header">
            <div style="margin-bottom: 0.5rem;">
                <span class="badge-pill badge-dsp">Deterministic DSP</span>
                <span class="badge-pill badge-zero-ml">Zero ML Models</span>
                <span class="badge-pill badge-16k">Canonical 16 kHz</span>
            </div>
            <h1>SignalGuard · Audio Quality & Forensics</h1>
            <p>A deterministic signal-processing quality-control layer for ML speech pipelines.
            Flag synthetic voices via physical acoustics and repair human speech distortions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_banner(result: PipelineResult) -> None:
    """Render the executive outcome card with distinct color-coded styling."""
    decision_styles = {
        PipelineDecision.PASS: (
            "decision-pass",
            "  PASS — CLEAN HUMAN SPEECH",
            "Audio passed all forensic heuristics and signal quality gates without requiring restoration.",
        ),
        PipelineDecision.PASS_RESTORED: (
            "decision-restored",
            "  PASS (RESTORED) — DISTORTIONS REPAIRED",
            "Hardware/environmental corruptions detected and successfully corrected by targeted DSP filters with measurable quality gain.",
        ),
        PipelineDecision.REJECT_LIKELY_SYNTHETIC: (
            "decision-reject-synth",
            "  REJECT — LIKELY AI-GENERATED / SYNTHETIC",
            "Signal exhibits multiple physical/acoustic anomalies characteristic of neural vocoders or TTS systems.",
        ),
        PipelineDecision.REJECT_UNRECOVERABLE: (
            "decision-reject-unrec",
            "  REJECT — UNRECOVERABLE CORRUPTION",
            "Audio has severe distortions that could not be restored within objective quality thresholds.",
        ),
    }
    css_class, title_text, desc_text = decision_styles.get(
        result.decision,
        ("decision-pass", result.decision.value, ""),
    )

    st.markdown(
        f"""
        <div class="decision-banner {css_class}">
            <h3 style="margin: 0; font-size: 1.3rem;">{title_text}</h3>
            <p style="margin: 0.3rem 0 0 0; font-size: 0.95rem;">{desc_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gauge_meter(score: float) -> None:
    """Render the synthetic evidence score gauge and risk classification bar."""
    synth_score = float(score)
    if synth_score >= 70.0:
        bar_color = "#ef4444"
        risk_label = "High Risk (Likely Synthetic)"
    elif synth_score >= 40.0:
        bar_color = "#f59e0b"
        risk_label = "Moderate Risk (Ambiguous / Borderline)"
    else:
        bar_color = "#10b981"
        risk_label = "Low Risk (Natural Characteristics)"

    st.markdown(
        f"""
        <div class="gauge-container">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600; font-size: 0.95rem;">Synthetic Evidence Index</span>
                <span style="font-weight: 700; color: {bar_color}; font-size: 1rem;">{synth_score:.1f}% · {risk_label}</span>
            </div>
            <div class="gauge-bar-bg">
                <div class="gauge-bar-fill" style="width: {min(100.0, max(2.0, synth_score))}%; background: {bar_color};"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8;">
                <span>0% (Natural Speech)</span>
                <span>40% (Borderline)</span>
                <span>70% (Synthetic Threshold)</span>
                <span>100% (High Anomaly)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_forensics_tab(result: PipelineResult) -> None:
    """Render Tab 1: Multi-domain forensic evidence scores, benchmarks, and rule explanations."""
    st.subheader("Forensic Domain Scores")
    domain_cols = st.columns(len(result.evidence.domain_scores))
    for col, domain in zip(domain_cols, result.evidence.domain_scores):
        with col:
            st.markdown(
                f"""
                <div class="card-stat">
                    <div class="val">{domain.score:.1f}</div>
                    <div class="lbl">{domain.name.replace('_', ' ')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("#### Heuristic Rule Evaluations")
    eval_rows = []
    for domain in result.evidence.domain_scores:
        eval_rows.append({
            "Domain": domain.name.title(),
            "Score (0-100)": round(domain.score, 2),
            "Triggered Rules": ", ".join(domain.triggered_rules) if domain.triggered_rules else "None (Pass)",
            "Status": "⚠️ Flagged" if domain.triggered_rules else "✅ Normal",
        })
    st.dataframe(eval_rows, width="stretch", hide_index=True)

    # Detailed benchmark table comparing observed metrics to typical human range
    st.markdown("#### Observed Forensic Acoustic Descriptors")
    observed_data = []

    f0_res = result.forensic_results.get("f0")
    if isinstance(f0_res, F0AnalysisResult) and f0_res.metrics:
        m = f0_res.metrics
        std_val = m.std_f0_hz
        rng_val = m.f0_range_hz
        observed_data.append({
            "Metric": "F0 Standard Deviation",
            "Observed": f"{std_val:.1f} Hz" if std_val is not None else "N/A",
            "Human Baseline": "10.0 – 50.0 Hz",
            "Synthetic Warning": "< 8.0 Hz (Monotone/Flat)",
        })
        observed_data.append({
            "Metric": "F0 Pitch Range",
            "Observed": f"{rng_val:.1f} Hz" if rng_val is not None else "N/A",
            "Human Baseline": "> 30.0 Hz",
            "Synthetic Warning": "< 20.0 Hz (Narrow Range)",
        })

    spec_res = result.forensic_results.get("spectral")
    if isinstance(spec_res, AnalysisResult) and spec_res.metrics:
        sm = spec_res.metrics
        flat = sm.get("mean_spectral_flatness")
        zcr = sm.get("mean_zero_crossing_rate")
        cent_cv = sm.get("centroid_cv")
        observed_data.append({
            "Metric": "Mean Spectral Flatness",
            "Observed": f"{flat:.4f}" if flat is not None else "N/A",
            "Human Baseline": "0.10 – 0.30",
            "Synthetic Warning": "> 0.50 (Noise-like spectrum)",
        })
        observed_data.append({
            "Metric": "Mean Zero Crossing Rate (ZCR)",
            "Observed": f"{zcr:.4f}" if zcr is not None else "N/A",
            "Human Baseline": "0.01 – 0.05",
            "Synthetic Warning": "> 0.08 (Noisy / High ZCR)",
        })
        observed_data.append({
            "Metric": "Spectral Centroid Variation (CV)",
            "Observed": f"{cent_cv:.4f}" if cent_cv is not None else "N/A",
            "Human Baseline": "> 0.15 (Prosodic shifts)",
            "Synthetic Warning": "< 0.08 (Rigid Timbre)",
        })

    lpc_res = result.forensic_results.get("lpc")
    if isinstance(lpc_res, AnalysisResult) and lpc_res.metrics:
        lm = lpc_res.metrics
        kurt = lm.get("residual_kurtosis")
        gain = lm.get("prediction_gain_db")
        observed_data.append({
            "Metric": "LPC Residual Kurtosis",
            "Observed": f"{kurt:.2f}" if kurt is not None else "N/A",
            "Human Baseline": "> 5.0 (Impulsive glottal pulses)",
            "Synthetic Warning": "< 3.0 (Smeared vocoder pulses)",
        })
        observed_data.append({
            "Metric": "LPC Prediction Gain",
            "Observed": f"{gain:.1f} dB" if gain is not None else "N/A",
            "Human Baseline": "> 5.0 dB",
            "Synthetic Warning": "< 3.0 dB (Weak vocal resonances)",
        })

    bisp_res = result.forensic_results.get("bispectrum")
    if isinstance(bisp_res, AnalysisResult) and bisp_res.metrics:
        bm = bisp_res.metrics
        bic = bm.get("mean_bicoherence")
        max_bic = bm.get("max_bicoherence")
        observed_data.append({
            "Metric": "Mean Bicoherence",
            "Observed": f"{bic:.4f}" if bic is not None else "N/A",
            "Human Baseline": "> 0.15 (Non-linear coupling)",
            "Synthetic Warning": "< 0.08 (Decoupled harmonics)",
        })
        observed_data.append({
            "Metric": "Max Quadratic Phase Coupling",
            "Observed": f"{max_bic:.4f}" if max_bic is not None else "N/A",
            "Human Baseline": "> 0.40",
            "Synthetic Warning": "< 0.35 (Weak harmonic phase coupling)",
        })

    mod_res = result.forensic_results.get("modulation")
    if isinstance(mod_res, AnalysisResult) and mod_res.metrics:
        mm = mod_res.metrics
        s_ratio = mm.get("syllabic_energy_ratio")
        depth = mm.get("modulation_depth")
        observed_data.append({
            "Metric": "Modulation Syllabic Ratio (2–8 Hz)",
            "Observed": f"{s_ratio:.3f}" if s_ratio is not None else "N/A",
            "Human Baseline": "0.30 – 0.70",
            "Synthetic Warning": "Abnormal ratio (Robotic cadence)",
        })
        observed_data.append({
            "Metric": "Temporal Envelope Modulation Depth",
            "Observed": f"{depth:.3f}" if depth is not None else "N/A",
            "Human Baseline": "> 0.35",
            "Synthetic Warning": "< 0.20 (Flattened dynamic envelope)",
        })

    breath_res = result.forensic_results.get("breath")
    if isinstance(breath_res, AnalysisResult) and breath_res.metrics:
        brm = breath_res.metrics
        burst = brm.get("max_speech_burst_seconds")
        dead_ratio = brm.get("digital_silence_ratio")
        observed_data.append({
            "Metric": "Max Continuous Speech Burst",
            "Observed": f"{burst:.1f} s" if burst is not None else "N/A",
            "Human Baseline": "< 4.0 s (Natural breathing pauses)",
            "Synthetic Warning": "> 4.5 s (Unnatural run-on speech)",
        })
        observed_data.append({
            "Metric": "Digital Silence Gap Ratio",
            "Observed": f"{dead_ratio * 100:.1f}%" if dead_ratio is not None else "N/A",
            "Human Baseline": "0.0% (Ambient presence preserved)",
            "Synthetic Warning": "> 20.0% (Dead digital silence dropouts)",
        })

    decay_res = result.forensic_results.get("decay")
    if isinstance(decay_res, AnalysisResult) and decay_res.metrics:
        dcm = decay_res.metrics
        min_decay = dcm.get("min_offset_decay_ms")
        abrupt_cnt = dcm.get("abrupt_cutoff_count")
        observed_data.append({
            "Metric": "Utterance Offset Decay Duration",
            "Observed": f"{min_decay:.1f} ms" if min_decay is not None else "N/A",
            "Human Baseline": "15.0 – 60.0 ms (Room reverberation)",
            "Synthetic Warning": "< 8.0 ms (Step-function cut-off)",
        })
        observed_data.append({
            "Metric": "Abrupt Cut-Off Events",
            "Observed": f"{abrupt_cnt}" if abrupt_cnt is not None else "N/A",
            "Human Baseline": "0",
            "Synthetic Warning": "> 0 (Unnatural audio truncations)",
        })

    if observed_data:
        st.dataframe(observed_data, width="stretch", hide_index=True)

    st.markdown("#### Detailed Acoustic Explanations")
    if result.evidence.explanations:
        domain_meta: dict[str, tuple[str, str, str]] = {
            "f0": (
                "📈",
                "F0 Pitch & Prosody",
                "Evaluates fundamental pitch contour dynamics, prosodic vibrato, and inflection variation against human baseline.",
            ),
            "harmonic": (
                "🎵",
                "Harmonic Structure & HNR",
                "Inspects harmonic energy distribution and harmonics-to-noise ratio for vocoder buzzing or comb artifacts.",
            ),
            "spectral": (
                "📊",
                "Spectral Entropy & Timbre",
                "Monitors spectral centroid variation, spectral flatness, and Wiener entropy across frequency bands.",
            ),
            "phase": (
                "⚡",
                "Phase Continuity",
                "Measures unwrapped phase jumps and group delay deviations across time-frequency bins.",
            ),
            "temporal": (
                "⏱️",
                "Temporal Consistency",
                "Detects unnatural cross-segment uniformity in speech energy, pitch, and timbre across clauses.",
            ),
            "lpc": (
                "🎙️",
                "LPC Glottal Residual",
                "Analyzes glottal excitation pulse kurtosis and prediction gain from inverse vocal tract filtering.",
            ),
            "bispectrum": (
                "🔗",
                "Bispectral Phase Coupling",
                "Measures non-linear quadratic phase coupling (QPC) between vocal tract harmonic frequencies.",
            ),
            "modulation": (
                "🎚️",
                "Syllabic Modulation (2–8 Hz)",
                "Analyzes temporal amplitude envelope modulation depth and checks for rigid robotic rhythm peaks.",
            ),
            "breath": (
                "🫁",
                "Respiration & Pause Dynamics",
                "Detects unnatural continuous run-on speech and mathematically dead digital silence in speech gaps.",
            ),
            "decay": (
                "📉",
                "Reverberation & Offset Decay",
                "Evaluates physical room reverberation tails and flags abrupt step-function window truncations.",
            ),
        }
        for explanation in result.evidence.explanations:
            if ": " in explanation:
                domain_key, message = explanation.split(": ", 1)
            else:
                domain_key, message = "anomaly", explanation

            icon, label, context = domain_meta.get(
                domain_key.lower(),
                (
                    "⚠️",
                    domain_key.upper(),
                    "Acoustic anomaly detected by deterministic forensic rule comparison.",
                ),
            )
            st.markdown(
                f"""
                <div class="anomaly-card">
                    <div class="anomaly-header">
                        <span class="anomaly-badge">{icon} {label}</span>
                        <span style="font-size: 0.75rem; color: #f87171; font-weight: 700;">⚠️ THRESHOLD EXCEEDED</span>
                    </div>
                    <div class="anomaly-title">{message}</div>
                    <p class="anomaly-context"><em>Acoustic Significance:</em> {context}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="clean-evidence-card">
                <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.3rem;">
                    ✅ Clean Organic Speech Baseline
                </div>
                <div style="font-size: 0.9rem; line-height: 1.4; color: #d1fae5;">
                    No acoustic anomalies crossed synthetic thresholds. All analyzed physical domains—including
                    pitch contour variation, glottal pulse kurtosis, non-linear harmonic coupling, respiratory pauses,
                    and room reverberation decay—exhibit natural human bio-acoustic properties.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_restoration_tab(result: PipelineResult) -> None:
    """Render Tab 2: Detected corruptions, applied filters, and objective quality gates."""
    st.subheader("Signal Corruption Analysis")
    detections = result.corruption_before.to_dict()["detections"]

    if result.corruption_before.detected_kinds:
        st.markdown(
            "**Corruptions Detected**: "
            + ", ".join(f"`{k}`" for k in result.corruption_before.detected_kinds)
        )
    else:
        st.markdown(
            "<div style='color: #10b981; font-weight: 500; margin-bottom: 0.5rem;'>"
            "✅ No signal corruptions detected. Audio meets high-fidelity recording baseline."
            "</div>",
            unsafe_allow_html=True,
        )

    st.dataframe(detections, width="stretch")

    if result.applied_restorations:
        st.markdown("#### Applied DSP Restoration Filters")
        st.info(
            "Targeted filters applied in sequence: "
            + ", ".join(f"**{r}**" for r in result.applied_restorations)
        )

    if result.quality is not None:
        st.markdown("#### Objective Restoration Quality Gate")
        q_dict = result.quality.to_dict()
        if q_dict["accepted"]:
            status_text = "✅ ACCEPTED (Restoration Improved Signal Quality)"
        elif result.decision is PipelineDecision.PASS:
            status_text = (
                "⚠️ BYPASSED (Restoration Did Not Meet Quality Gain Threshold — Original Signal Preserved)"
            )
        else:
            status_text = "❌ REJECTED (Distortion Unrecoverable Within Objective Quality Thresholds)"
        st.markdown(f"**Quality Gate Decision**: {status_text}")
        st.dataframe(q_dict["checks"], width="stretch")

    if result.restored_samples is not None:
        st.markdown("#### Restored Audio Playback & Export")
        restored = np.asarray(result.restored_samples, dtype=np.float32)
        buffer = BytesIO()
        sf.write(
            buffer,
            restored,
            result.audio.sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
        st.audio(buffer.getvalue(), format="audio/wav")
        st.download_button(
            "Download Restored WAV",
            buffer.getvalue(),
            "signalguard_restored.wav",
            "audio/wav",
        )


def render_visuals_tab(result: PipelineResult) -> None:
    """Render Tab 3: Waveforms, STFT spectrograms, and before/after comparisons."""
    st.subheader("Spectral & Waveform Visualizations")

    if result.restored_samples is not None:
        restored = np.asarray(result.restored_samples, dtype=np.float32)
        st.markdown("#### Before vs. After Restoration Waveforms")
        st.pyplot(
            plot_before_after(
                result.audio.samples,
                restored,
                result.audio.sample_rate,
            )
        )

        st.markdown("#### Before vs. After Restoration Spectrograms")
        st.pyplot(
            plot_before_after_spectrogram(
                result.audio.samples,
                restored,
                result.audio.sample_rate,
            )
        )
    else:
        wave_col, spec_col = st.columns(2)
        with wave_col:
            st.markdown("#### Canonical Input Waveform")
            st.pyplot(
                plot_waveform(
                    result.audio.samples,
                    result.audio.sample_rate,
                    title="Canonical Input Waveform (16 kHz)",
                )
            )
        with spec_col:
            st.markdown("#### STFT Spectrogram (dB Scale)")
            st.pyplot(
                plot_spectrogram(
                    result.audio.samples,
                    result.audio.sample_rate,
                    title="Magnitude Spectrogram (0 – 8,000 Hz)",
                )
            )


def render_pipeline_flow_tab(result: PipelineResult) -> None:
    """Render Tab 4: Step-by-step pipeline execution path."""
    st.subheader("Pipeline Execution Path")
    st.caption("Trace of deterministic decisions executed for this recording:")

    is_synth = result.evidence.decision is EvidenceDecision.LIKELY_SYNTHETIC
    has_corruption = bool(result.corruption_before.detected_kinds)
    is_restored = result.decision is PipelineDecision.PASS_RESTORED

    step1_badge = "✅ COMPLETED"
    step2_badge = "✅ COMPLETED"
    step3_badge = f"  FLAGGED ({result.evidence.score:.1f}%)" if is_synth else f"  PASSED ({result.evidence.score:.1f}%)"
    step4_badge = "SKIPPED (Synthetic Halt)" if is_synth else ("⚠️ DETECTED" if has_corruption else "✅ CLEAN")
    step5_badge = ("✅ APPLIED (" + ", ".join(result.applied_restorations) + ")") if (not is_synth and has_corruption) else "SKIPPED"
    step6_badge = ("✅ PASSED" if is_restored else "  REJECTED") if (not is_synth and has_corruption) else "SKIPPED"
    step7_badge = f"🎯 {result.decision.value}"

    st.markdown(
        f"""
        | Step | Operation | Execution Outcome |
        | :--- | :--- | :--- |
        | **1. Preprocessing** | Mono conversion, 16 kHz resample, DC removal, peak limit | `{step1_badge}` |
        | **2. Segmentation** | Lossless 3.0s boundary chunking | `{step2_badge}` |
        | **3. AI Forensics** | Multi-domain evidence scoring (F0, Harmonic, Spectral, Phase) | `{step3_badge}` |
        | **4. Quality Check** | Detection of noise, hum, clipping, impulses, dropouts | `{step4_badge}` |
        | **5. DSP Restoration** | Targeted Wiener, Notch, Median, or Interpolation filters | `{step5_badge}` |
        | **6. Quality Gate** | Objective post-repair SNR improvement gate | `{step6_badge}` |
        | **7. Final Decision** | Pipeline decision outcome | **`{step7_badge}`** |
        """,
    )
