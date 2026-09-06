# SignalGuard

SignalGuard is a deterministic, explainable audio forensics and restoration SDK designed for speech-oriented processing pipelines. It relies purely on classical Digital Signal Processing (DSP); it does **not** train or run black-box machine learning models or neural networks.

> **Forensic Disclaimer**: SignalGuard does not claim to definitively prove provenance or authenticate audio. Its **Synthetic Evidence Index** is a transparent, calibrated collection of deterministic physical cues grounded in vocal biomechanics and acoustics that assists analysts and automated data quality systems.

---

## Processing Pipeline

```text
Audio Input -> Preprocessing (16 kHz Mono, Safe Resampling, DC Removal, Peak Norm)
            -> Boundary-Safe Windowed Segmentation
            -> 10 Parallel Forensic Analysis Channels
            -> Multi-Domain Corroboration Fusion -> Synthetic Evidence Score (0–100)
            │
            ├── If Evidence >= Threshold: REJECT_LIKELY_SYNTHETIC
            │
            └── If Evidence < Threshold (Natural Speech):
                  │
                  ├── Corruption Detector -> If Clean / No Distortions: PASS
                  │
                  └── If Hardware / Environmental Distortions Detected:
                        -> Apply Targeted Deterministic DSP Restoration Filters
                        -> Objective Quality Gate (RMS bounds, SNR gain, artifact check)
                              ├── Quality Accepted: PASS_RESTORED
                              └── Quality Rejected:
                                    ├── Severe Distortion: REJECT_UNRECOVERABLE
                                    └── Minor / False Alarm: PASS (Bypassed, Raw Audio Preserved)
```

---

## Core Algorithms Explained

SignalGuard extracts physical acoustic invariants across **10 forensic domains** and couples them with targeted, reversible DSP restoration algorithms. Below is an overview of each algorithm, combining rigorous mathematical terminology with intuitive physical explanations.

### 1. Forensic Detection Algorithms (AI vs. Human Voice)

#### 🎙️ YIN Fundamental Frequency ($F_0$) & Micro-Prosody Tracking
* **The Concept**: Human speech pitch continuously fluctuates due to subglottal lung pressure changes, emotional inflection, and muscular micro-tremors (jitter). Neural text-to-speech (TTS) vocoders frequently produce pitch contours that are either unnaturally flat, mechanically quantized, or lack natural micro-inflections.
* **How It Works**: Calculates the cumulative mean normalized difference function of the signal autocorrelation to pinpoint the exact pitch period without octave errors, followed by parabolic interpolation on local minima. The system computes the voiced frame ratio, pitch standard deviation, octave range, and first/second-order delta velocity to identify robotic pitch constancy.

#### 🫁 Linear Predictive Coding (LPC) Glottal Residual Analysis
* **The Concept**: Speech acoustics can be modeled as an acoustic source (vocal folds vibrating at the glottis) filtered by a resonant filter (the vocal tract, mouth, and nasal cavities creating formants). Human glottal pulses are asymmetric and impulsive, producing non-Gaussian residual bursts when the vocal tract resonance is inverse-filtered out. Neural vocoders generate audio through recurrent or diffusion networks that often leave overly smooth, Gaussianized, or inconsistent excitation residuals.
* **How It Works**: Applies Levinson-Durbin all-pole deconvolution ($p=16$ poles) to model the vocal tract filter $A(z)$, then inverts the filter to isolate the raw glottal excitation residual $e[n] = x[n] - \sum a_k x[n-k]$. It then calculates residual excess kurtosis and frame variance dynamics.

#### 🔗 Higher-Order Spectral Analysis: Bispectrum Phase Coupling
* **The Concept**: The human vocal apparatus is an inherently non-linear biomechanical system. When vocal cords oscillate at different frequencies ($f_1$ and $f_2$), non-linear tissue interactions generate sum and difference harmonics ($f_3 = f_1 + f_2$) whose phases are tightly coupled (Quadratic Phase Coupling). Standard neural vocoders produce frequencies additively or through separate harmonic generators, failing to reproduce genuine biological phase coupling.
* **How It Works**: Computes the third-order cumulant spectrum (the Bispectrum):
  $$B(f_1, f_2) = \mathbb{E}\left[X(f_1) X(f_2) X^*(f_1 + f_2)\right]$$
  and derives normalized bicoherence. Elevated bicoherence indicates authentic biological phase coupling, while near-zero bicoherence indicates synthetic multi-tone synthesis.

#### 🎚️ Syllabic Modulation Spectrum (2–8 Hz Envelope Dynamics)
* **The Concept**: Human speech rhythm is governed by the opening and closing of the jaw and lips, concentrating syllable production energy into a very specific low-frequency modulation band (2 to 8 Hz). Synthetic TTS often exhibits unnaturally continuous rhythm, missing the distinct, peaked syllabic envelope rhythms of live human articulation.
* **How It Works**: Decomposes the speech signal into octave sub-bands, computes the instantaneous Hilbert envelope of each band, and performs a secondary Fourier transform to inspect the envelope spectrum. It measures the ratio of modulation power inside the 2–8 Hz window relative to total envelope energy.

#### 🌬️ Respiration & Pre-Phonatory Breath Dynamics
* **The Concept**: Natural speakers must periodically inhale before speaking and during conversational pauses. Inhaling produces a distinct acoustic signature: a low-amplitude, high-frequency turbulence ramp preceding energy onset. AI voice generators either skip breathing entirely, place abrupt dead silence between phrases, or paste unnatural, repetitive breath samples.
* **How It Works**: Analyzes energy transitions and spectral centroid shifts within pause-to-speech boundaries. Pre-phonatory breath segments are identified by elevated high-frequency spectral centroids coupled with gradual energy growth preceding voiced speech.

#### 📉 Reverberant Tail & Phoneme Offset Decay
* **The Concept**: When a real human stops speaking a word or vowel, the sound does not instantly vanish; it decays following the acoustic relaxation of the vocal tract and room reverberation ($T_{60}$ exponential energy dissipation). Neural vocoders operating frame-by-frame often gate audio abruptly, producing unnatural sharp cutoff edges or synthetic silence drops.
* **How It Works**: Detects voicing offsets and fits a logarithmic decay model ($E(t) = E_0 e^{-\lambda t}$) to the trailing 100 ms of phonemes. Unnaturally high decay rates ($\lambda > 50\text{ s}^{-1}$) or instantaneous zero-crossings reveal synthetic gating.

#### 🎵 Harmonic Comb Filtering & Harmonic-to-Noise Ratio (HNR)
* **The Concept**: Healthy voiced speech features clear integer harmonics ($F_0, 2F_0, 3F_0, \dots$) generated by the periodic clapping of the vocal folds, surrounded by low baseline aspiration noise. Vocoder artifacts often exhibit smeared or inharmonic sidebands and abnormal harmonic energy proportions.
* **How It Works**: Constructs a pitch-locked comb filter centered on the instantaneous $F_0$, measures the energy within harmonic bins versus inter-harmonic valleys, and calculates the true Harmonic-to-Noise Ratio (HNR) in decibels.

#### 📊 Spectral Timbre Stability & Entropy
* **The Concept**: During continuous human speech, the tongue and vocal tract constantly move, causing continuous shifts in spectral centroid, spectral flux, and spectral flatness. Synthetic speech often exhibits unvarying, frozen spectral envelopes across successive vowels.
* **How It Works**: Evaluates frame-by-frame Short-Time Fourier Transform (STFT) descriptors, including Wiener entropy (spectral flatness), Shannon spectral entropy, spectral rolloff (85%), and spectral flux coefficient of variation across voiced segments.

#### ⚡ STFT Phase Continuity & Group Delay
* **The Concept**: When humans speak, phase progresses smoothly and predictably between overlapping analysis frames according to instantaneous frequency. Neural vocoders that reconstruct time-domain audio from magnitude spectrograms (e.g., Griffin-Lim, HiFi-GAN, MelGAN) often introduce discontinuous phase jumps or dispersed group delay spikes.
* **How It Works**: Computes the expected inter-frame phase advancement $\Delta \phi = 2\pi f \Delta t$, measures the angular difference between actual and expected phase, and calculates group delay variance $\tau_g(\omega) = -d\phi/d\omega$.

#### ⏱️ Multi-Segment Temporal Consistency
* **The Concept**: Across a sentence or conversation, human speech exhibits natural macroscopic dynamic range variance and pauses. AI generators often enforce excessive global normalization or artificial uniformity across segmented intervals.
* **How It Works**: Segments audio into non-overlapping windows (1.5–3.0 s) and calculates cross-segment coefficients of variation (CV) for RMS energy, zero-crossing rate (ZCR), and spectral centroid.

#### ⚖️ Corroboration Fusion & Evidence Index
* **The Concept**: A single metric can be influenced by audio quality or recording gear; true synthetic voice generation exhibits cross-domain anomalies.
* **How It Works**: Assigns domain weights across all 10 acoustic channels (summing to 100%). It then applies non-linear corroboration synergy: when multiple independent domains simultaneously detect strong synthetic cues, an exponential corroboration multiplier elevates the final score toward 90–100%, while clean speech with isolated artifacts remains safely below the decision threshold.

---

### 2. Corruption Detection & Targeted DSP Restoration Algorithms

SignalGuard separates **synthetic voice artifacts** from **hardware and environmental corruptions**, restoring real speech without modifying genuine vocal dynamics.

* **Broadband Noise (Stationarity-Quantile SNR & Spectral Subtraction)**:
  Estimates background noise power using low-energy stationarity quantiles. When broadband noise is detected, it applies adaptive spectral magnitude subtraction:
  $$|\hat{S}(f)| = \max\left(|Y(f)| - \alpha |N(f)|, \beta |Y(f)|\right)$$
  using an over-subtraction factor $\alpha$ and spectral floor parameter $\beta$ to eliminate musical noise.
* **Mains Hum (Harmonic Comb Notch Filtering)**:
  Identifies AC electrical mains interference at 50 Hz or 60 Hz. To prevent false positives on vocal pitch harmonics (e.g. 100 Hz male voices), it strictly validates that the fundamental mains frequency ($h=1$) is elevated before applying a cascade of high-$Q$ digital IIR notch filters across mains harmonics.
* **Impulses & Clicks (Median Absolute Deviation Excision)**:
  Identifies isolated electrical pops and microphone clicks using Median Absolute Deviation (MAD) outlier detection on local residual amplitudes. Flagged impulses are excised and smoothly reconstructed via context-aware polynomial interpolation.
* **Dropouts (Near-Zero Boundary Interpolation)**:
  Scans for packet-loss dropouts or digital transmission mutes ($\le -80\text{ dBFS}$) longer than 20 ms, reconstructing lost samples using linear phase-matching boundary interpolation.
* **Objective Quality Gate**:
  Every restored signal must pass an objective verification gate before being accepted: absolute RMS change must not exceed $\pm 3\text{ dB}$, post-processing clipping must not increase, and signal-to-noise ratio must measurably improve. If a restoration filter degrades the audio or is unnecessary, SignalGuard safely bypasses the filter and passes the original clean signal.

---

## Operational Repository Structure

Below is the directory structure highlighting the essential operational files:

```text
SignalGuard/
├── config.yaml                     # Centralized configuration: thresholds, weights, and DSP parameters
├── requirements.txt                # Production and development dependencies (NumPy, SciPy, Streamlit, etc.)
│
├── src/                            # Core SignalGuard SDK and engine
│   ├── pipeline.py                 # End-to-end orchestrator: preprocessing, analysis, restoration, decision logic
│   ├── preprocessing.py            # 16 kHz mono conversion, resampling, DC removal, peak limiting, segmentation
│   ├── scoring.py                  # Multi-domain evidence aggregation, corroboration fusion, decision thresholds
│   ├── quality.py                  # Post-restoration objective quality gate (RMS bounds, SNR gain, checks)
│   ├── calibration.py              # Empirical threshold calibration tool to find F1-optimal synthetic boundaries
│   ├── visualization.py            # Headless-safe Matplotlib waveform, spectrogram, and before/after visualizers
│   │
│   ├── detectors/                  # Forensic analysis and corruption detection modules
│   │   ├── f0_analysis.py          # YIN pitch tracking, micro-prosodic dynamics, jitter, and pitch contour
│   │   ├── lpc_analysis.py         # Linear Predictive Coding (LPC) vocal tract deconvolution & glottal residual
│   │   ├── bispectrum_analysis.py  # Higher-Order Spectral Analysis (HOSA) detecting Quadratic Phase Coupling
│   │   ├── modulation_analysis.py  # Syllabic envelope dynamics (2–8 Hz modulation spectrum energy ratio)
│   │   ├── breath_analysis.py      # Respiration dynamics, pre-speech inhalations, and pause boundary detection
│   │   ├── decay_analysis.py       # Phoneme offset energy decay rates and reverberation tail continuity
│   │   ├── harmonic_analysis.py    # Harmonic-to-Noise Ratio (HNR) and pitch-locked harmonic comb extraction
│   │   ├── spectral_analysis.py    # Spectral entropy, flatness, centroid uniformity, and timbre stability
│   │   ├── phase_analysis.py       # Inter-frame STFT phase jump statistics and group delay smoothness
│   │   ├── temporal_analysis.py    # Cross-segment variance of speech descriptors over time
│   │   └── corruption_detection.py # Detection of mains hum (fundamental-verified), clipping, dropouts, noise
│   │
│   ├── restoration/                # Deterministic signal restoration
│   │   └── filters.py              # Targeted DSP filters: spectral subtraction, comb notches, click excision
│   │
│   ├── ui/                         # Modular Streamlit dashboard
│   │   ├── views.py                # Executive decision banner, glassmorphic anomaly cards, diagnostics tables
│   │   ├── styles.py               # Glassmorphic CSS design system, dark gradients, badges, hover effects
│   │   ├── sidebar.py              # Audio input selectors, configuration summary, and sensitivity tuning
│   │   └── presets.py              # Synthetic benchmark audio presets for instant live demonstration
│   │
│   ├── cli.py                      # Unified command-line interface (analyze, evaluate, calibrate, generate-data)
│   ├── dataset_generator.py        # Benchmark audio synthesizer generating clean, synthetic, and corrupted files
│   ├── evaluate.py                 # Benchmark evaluation matrix, confusion metrics, and calibration report
│   └── streamlit_app.py            # Local Streamlit inspection dashboard runner
│
├── tests/                          # Automated test suite (148 tests, 100% deterministic)
│   ├── test_new_detectors.py       # Unit tests for LPC residual, Bispectrum, and Modulation Spectrum
│   ├── test_breath_and_decay.py    # Unit tests for Respiration and Reverberation Decay analyzers
│   ├── test_regressions.py         # Cross-module regression tests for pipeline decisions and hum fixes
│   └── ...                         # Dedicated tests for calibration, preprocessing, restoration, and UI
```

---

## Installation & Setup

SignalGuard targets **Python 3.11** and requires only standard open-source scientific Python packages.

```powershell
# 1. Create and activate a virtual environment:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Upgrade pip and install dependencies:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## SDK Usage

Analyze an audio recording in five lines of code:

```python
from src.pipeline import SignalGuardPipeline

# Initialize the pipeline (loads thresholds from config.yaml)
pipeline = SignalGuardPipeline()

# Run full forensic analysis and restoration evaluation
result = pipeline.analyze_file("recording.wav")

# Inspect the executive outcome:
print("Pipeline Decision  :", result.decision.value)         # PASS, PASS_RESTORED, REJECT_LIKELY_SYNTHETIC, or REJECT_UNRECOVERABLE
print("Synthetic Evidence :", result.evidence.score, "/ 100") # Weighted forensic risk index
print("Risk Category      :", result.evidence.decision.value) # CLEAN_HUMAN, BORDERLINE_REVIEW, or LIKELY_SYNTHETIC
print("Corruptions Found  :", result.corruption_before.detected_kinds)

# Access restored audio samples if repairs were accepted:
if result.restored_samples is not None:
    print("Applied DSP Filters:", result.applied_restorations)
```

You can also run individual forensic detectors directly:

```python
from src.detectors import analyze_f0, analyze_lpc, analyze_bispectrum

pitch = analyze_f0(samples, 16_000)
lpc_residual = analyze_lpc(samples, 16_000)
phase_coupling = analyze_bispectrum(samples, 16_000)
```

---

## Unified Command-Line Interface (CLI)

SignalGuard features a built-in CLI for analysis, benchmark generation, and evaluation:

```powershell
# 1. Analyze an audio file:
python -m src.cli analyze path/to/speech.wav
python -m src.cli analyze path/to/speech.wav --json

# 2. Generate benchmark synthetic and corrupted datasets into data/:
python -m src.cli generate-data --count 10

# 3. Run complete evaluation benchmark across all dataset classes:
python -m src.cli evaluate

# 4. Calibrate the decision threshold on provenance-labelled folders:
python -m src.cli calibrate --natural-dir data/clean --synthetic-dir data/synthetic
```

---

## Local Web Dashboard (Streamlit)

Launch the interactive inspection interface:

```powershell
streamlit run src/streamlit_app.py
```

The application provides:
* **Executive Decision Banner**: Color-coded status badge (`PASS`, `PASS_RESTORED`, `REJECT`).
* **Synthetic Evidence Risk Gauge**: Visual dial displaying the composite forensic index.
* **Detailed Acoustic Explanations**: Glassmorphic cards highlighting triggered physical anomalies with domain badges and explanations.
* **Restoration Quality Gate**: Tabular breakdown of pre- and post-repair metrics, RMS changes, and filter acceptance.
* **Spectral Visualizations**: High-resolution Matplotlib waveforms, STFT spectrograms, and before-vs-after comparison figures.
* **Audio Player & Export**: In-browser audio playback with one-click download of restored 16-bit PCM WAVs.

---

## Running the Test Suite

SignalGuard includes a comprehensive test suite of 148 deterministic tests:

```powershell
python -m pytest -q -W error
```

All tests run locally in under 8 seconds without network access, GPU hardware, or external datasets.
