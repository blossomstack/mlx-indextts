"""Local, user-supplied-input benchmark. No downloads, bundled voices or uploads."""

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_indextts import IndexTTSv25

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model-dir", required=True)
p.add_argument("--reference", required=True)
p.add_argument("--text-file", type=Path, required=True)
p.add_argument("--language", choices=["zh", "en"], required=True)
p.add_argument("--backend", choices=["reference", "metal"], required=True)
p.add_argument("--trials", type=int, default=3)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.trials < 1:
    p.error("trials must be positive")
a.output.mkdir(parents=True, exist_ok=False)
mx.set_cache_limit(256 * 1024**2)
start = time.perf_counter()
e = IndexTTSv25(a.model_dir, reference_backend="mlx", vocoder_backend=a.backend, memory_limit_gb=6)
e.save_speaker(a.reference, str(a.output / "speaker.npz"))
e.release_preprocessing_models()
report = {
    "mlx": mx.__version__,
    "platform": platform.platform(),
    "backend": a.backend,
    "setup_and_reference_seconds": time.perf_counter() - start,
    "settings": {
        "seed": 2026,
        "diffusion_steps": 16,
        "max_mel_tokens": 650,
        "max_text_tokens_per_segment": 64,
        "interval_silence": 250,
        "emotion": "calm",
        "emo_alpha": 0.6,
    },
    "runs": [],
}
text = a.text_file.read_text()
for trial in range(a.trials + 1):
    mx.reset_peak_memory()
    start = time.perf_counter()
    chunks = list(
        e.stream(text, str(a.output / "speaker.npz"), language=a.language, **report["settings"])
    )
    audio = np.concatenate([c.audio for c in chunks])
    elapsed = time.perf_counter() - start
    assert np.isfinite(audio).all()
    assert all(np.array_equal(c.audio[-5512:], np.zeros(5512)) for c in chunks[:-1])
    row = {
        "warmup": trial == 0,
        "seconds": elapsed,
        "audio_seconds": len(audio) / 22050,
        "rtf": elapsed / (len(audio) / 22050),
        "peak_mlx_bytes": mx.get_peak_memory(),
        "segment_frames": [len(c.audio) for c in chunks],
        "waveform_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
    }
    report["runs"].append(row)
    if trial == 1:
        sf.write(a.output / "review.wav", audio, 22050)
        np.save(a.output / "waveform.npy", audio)
    print(row, flush=True)
report["median_seconds"] = statistics.median(
    r["seconds"] for r in report["runs"] if not r["warmup"]
)
report["forbidden_modules"] = [
    n for n in sys.modules if n.split(".")[0] in {"torch", "torchaudio", "transformers"}
]
assert not report["forbidden_modules"]
(a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
