# MLX IndexTTS

IndexTTS 2.5 inference on Apple Silicon, including native reference-audio conditioning and a specialized Metal BigVGAN vocoder. Mandarin and English have been exercised end to end; other upstream languages are not equally validated here.

This project builds on [vanch007/mlx-indextts2](https://github.com/vanch007/mlx-indextts2), version 0.2.0, baseline `a7666367b8551656a2029ad75f259cb5e4936b3b`. It is an independent implementation, not an official IndexTTS release. See [attribution and licenses](NOTICE.md).

## Native installation

Use an Apple Silicon Mac, macOS with Metal support, and Python 3.12. The measured configuration used MLX 0.31.1 and 16 GB unified memory.

```sh
git clone https://github.com/blossomstack/mlx-indextts.git
cd mlx-indextts
python3.12 -m venv .venv-native
.venv-native/bin/python -m pip install '.[native]'
```

The `native` extra does not require Torch, torchaudio or Transformers. Conversion and reference/parity tooling use separate extras and should use a separate environment. Explicitly select `reference_backend="mlx"`: the legacy default remains `"torch"` for compatibility. The existing CLI/web/API interfaces may use that legacy default; the Python API below is the validated native entry point.

## Prepare licensed model assets

Weights, voice recordings and speaker caches are **not distributed in this repository**. Review the applicable model terms before obtaining assets. IndexTTS model terms are distinct from this port's MIT license and also cover applicable upstream code/derivatives; see [NOTICE.md](NOTICE.md) and [the Bilibili agreement](licenses/LicenseRef-bilibili.txt).

Follow [asset conversion](docs/assets.md) to prepare a converted IndexTTS-2.5 directory, native CAMPPlus/reference statistics and W2V-BERT. Use only reference recordings you are authorized to use.

## Synthesize

```python
import numpy as np
import soundfile as sf
from mlx_indextts import IndexTTSv25

engine = IndexTTSv25(
    "models/mlx-v25-8bit",
    reference_backend="mlx",       # native audio features + W2V-BERT + CAMPPlus
    vocoder_backend="metal",       # use "reference" for the unfused MLX fallback
    memory_limit_gb=6,              # soft allocation limit, not a strict peak ceiling
)
chunks = list(engine.stream(
    "Hello. The garden is quiet this evening.",
    "reference.wav",
    language="en",                 # use "zh" for Mandarin
    emotion="calm",
    emo_alpha=0.6,
    seed=2026,
    diffusion_steps=16,
    max_text_tokens_per_segment=64,
    max_mel_tokens=650,
    interval_silence=250,
))
sf.write("speech.wav", np.concatenate([c.audio for c in chunks]), 22050)
```

To reuse a voice without recomputing conditioning, call `engine.save_speaker("reference.wav", "speaker.npz")`, then use `speaker.npz` in place of the WAV. `engine.release_preprocessing_models()` releases reference encoders after caching. Speaker caches contain voice-derived data; keep them private unless you have permission to distribute them.

## Performance and correctness

A local M2 Pro comparison of the same engine with reference versus Metal anti-aliasing measured:

| Generated audio | Reference inference | Metal inference | Peak MLX allocation |
|---|---:|---:|---:|
| Mandarin, 30.2 s | 26.40 s | 16.07 s | 5.70 → 4.67 GB |
| English, 24.4 s | 22.44 s | 12.95 s | 6.57 → 5.60 GB |

Three measured trials after warmup, fixed quality settings, sequential workloads. The vocoder was approximately 5× faster; total request time fell 39–42%. English includes variation in untouched stages: vocoder savings alone predict about 38% less total time. These are local observations, not hardware-independent guarantees. [Methodology and limitations](docs/benchmarking.md) includes reproduction instructions.

Semantic tokens, durations and inserted pauses matched exactly; waveform RMS differences were 1.0–2.4e-6. The same ASR transcripts before/after do not prove perceptual equivalence. Known Mandarin wording/transcription discrepancies remain unresolved. Long passages need segmentation and listening review. Emotion and speaker similarity are not guaranteed across every input.

## Tests and contributions

```sh
# Separate development environment on Apple Silicon.
python3.12 -m venv .venv-dev
.venv-dev/bin/python -m pip install '.[native,dev,api,webui]'
.venv-dev/bin/python -m pytest -q
```

Metal numerical tests require a real Metal device. Weight-dependent legacy tests skip when their fixtures are absent. Hosted CI runs portable model-contract tests, syntax/critical lint checks and package builds; it does not claim GPU or speech-quality validation. The clean source export was validated locally; no model assets are downloaded by CI.

Use pull requests for changes. Keep checkpoints, generated voices, credentials and local environment files out of commits. Dependencies and GitHub Actions receive routine Dependabot updates. No package or model release is published automatically.
