# SignalGuard

SignalGuard is a deterministic, explainable audio-forensics and restoration SDK
for speech-oriented data pipelines. It uses classical DSP only; it does not
train or run an ML classifier.

> SignalGuard does not prove that a recording is AI-generated or authenticate
> audio. Its Synthetic Evidence Score is a configurable collection of heuristic
> cues that requires calibration and human interpretation.

## Included

- Canonical mono, 16 kHz preprocessing with safe PCM conversion, resampling,
  DC removal, peak limiting, and lossless segmentation.
- YIN F0 analysis with explicit energy/periodicity voicing gates, voiced-ratio,
  mean, spread, range, first differences, and second-order changes.
- F0-guided harmonic energy/HNR, spectral shape/stability, STFT phase
  continuity, and cross-segment temporal consistency.
- Explainable, weighted Synthetic Evidence Score with per-rule explanations.
- Detection of broadband noise, hum, impulses, dropouts, rumble, hiss, and
  clipping; targeted deterministic repair and objective post-repair checks.
- Headless Matplotlib figures and a local Streamlit inspection app.

All operating thresholds, weights, and filter parameters are in
[`config.yaml`](config.yaml). Defaults are engineering baselines, not universal
forensic constants.

## Pipeline

```text
Input -> preprocess -> segment -> five forensic channels -> evidence score
      -> likely synthetic: REJECT_LIKELY_SYNTHETIC
      -> otherwise: corruption detection -> clean: PASS
         -> targeted restoration -> objective quality gate
            -> improved: PASS_RESTORED
            -> otherwise: REJECT_UNRECOVERABLE
```

## Setup

SignalGuard targets Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

WAV is the cross-platform baseline. Other formats, including MP3, depend on
the available `libsndfile` codecs and fail with a clear load error when absent.

## SDK usage

```python
from src.pipeline import SignalGuardPipeline

pipeline = SignalGuardPipeline()
result = pipeline.analyze_file("speech.wav")

print(result.decision.value)
print(result.evidence.score, result.evidence.decision.value)
print(result.corruption_before.detected_kinds)

if result.restored_samples is not None:
    print(result.applied_restorations)
```

Individual detectors are usable directly:

```python
from src.detectors import analyze_f0, analyze_harmonics, analyze_spectral

pitch = analyze_f0(samples, 16_000)
harmonics = analyze_harmonics(samples, 16_000, f0_result=pitch)
spectrum = analyze_spectral(samples, 16_000)
```

Each result exposes typed fields, threshold `rule_evaluations`, and a JSON-safe
`to_dict()` method. Insufficient audio or silence returns an explicit status
rather than fabricated evidence.

## Local UI

```powershell
streamlit run src/streamlit_app.py
```

The app displays the decision, evidence explanation, corruption indicators,
waveforms/spectrogram, post-restoration comparison, and a WAV download when a
repair is accepted. Select a file, preview it, and press **Analyze audio**.
Decode failures are shown directly in the page; PCM WAV is the recommended
baseline when a platform lacks optional compressed-audio codecs.

## Calibration

For production use, calibrate the score boundary with representative,
provenance-labelled recordings. Put natural and synthetic recordings in two
separate directories, reserve a held-out set for validation, and run:

```python
from src.calibration import calibrate_from_directories

report = calibrate_from_directories("data/clean", "data/synthetic")
print(report.recommended_likely_synthetic_threshold)
print(report.to_dict())
```

The toolkit scores every supplied file, selects a deterministic F1-optimal
`likely_synthetic_min` candidate (conservative tie-breaking), and reports its
confusion matrix, precision, recall, specificity, and F1. It never modifies
[`config.yaml`](config.yaml) automatically; review the report on a held-out set
before applying a threshold as policy. The repository does not ship a
representative labelled corpus, so it cannot honestly claim a universal
real-world calibrated value.

## Tests

```powershell
python -m pytest -q -W error
```

Tests synthesize tones, noise, hum, impulses, and dropouts in memory (plus
temporary WAVs). No external dataset, network, microphone, or trained model is
needed.

## Repository layout

```text
src/
|-- detectors/       # F0, harmonic, spectral, phase, temporal, corruption
|-- restoration/     # targeted deterministic DSP filters
|-- pipeline.py      # end-to-end decision orchestration
|-- scoring.py       # transparent weighted evidence aggregation
|-- quality.py       # post-restoration quality gate
|-- visualization.py # headless-safe Matplotlib figures
`-- streamlit_app.py # local inspection interface
```
