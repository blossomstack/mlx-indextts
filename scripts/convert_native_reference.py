"""Convert already-downloaded reference assets; no network or license acceptance."""

import argparse
from pathlib import Path
import mlx.core as mx
import torch
from mlx_indextts.models.campplus import convert_state_dict

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model-dir", type=Path, required=True)
p.add_argument("--campplus-checkpoint", type=Path, required=True)
a = p.parse_args()
paths = [a.model_dir / "reference-stats.safetensors", a.model_dir / "campplus.safetensors"]
if any(path.exists() for path in paths):
    p.error("Native reference output already exists; use a separate model directory.")
stats = torch.load(a.model_dir / "wav2vec2bert_stats.pt", map_location="cpu", weights_only=True)
weights = {
    "mean": mx.array(stats["mean"].detach().numpy()),
    "std": mx.array(torch.sqrt(stats["var"]).detach().numpy()),
    "emotion": mx.array(
        torch.load(a.model_dir / "feat2.pt", map_location="cpu", weights_only=True).detach().numpy()
    ),
    "speaker": mx.array(
        torch.load(a.model_dir / "feat1.pt", map_location="cpu", weights_only=True).detach().numpy()
    ),
}
camp = convert_state_dict(torch.load(a.campplus_checkpoint, map_location="cpu", weights_only=True))
mx.save_safetensors(str(paths[0]), weights)
mx.save_safetensors(str(paths[1]), camp)
print("Converted native reference assets. Original model licenses continue to apply.")
