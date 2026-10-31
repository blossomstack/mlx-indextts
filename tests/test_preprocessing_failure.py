"""A missing speaker checkpoint must never silently produce a random voice."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_campplus_download_failure_is_fatal(monkeypatch, tmp_path):
    import huggingface_hub
    import transformers
    from mlx_indextts.generate_v2 import IndexTTSv2
    from mlx_indextts.indextts.utils import maskgct_utils

    monkeypatch.setattr(maskgct_utils, "build_semantic_model", lambda **kw: (Mock(), Mock(), Mock()))
    monkeypatch.setattr(transformers.AutoFeatureExtractor, "from_pretrained", Mock())
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", Mock(side_effect=OSError("offline")))
    (tmp_path / "stats.pt").touch()
    tts = IndexTTSv2.__new__(IndexTTSv2)
    tts.mlx_model_dir = tts.model_dir = tmp_path
    tts.cfg = SimpleNamespace(w2v_stat="stats.pt")
    tts.device = "cpu"
    with pytest.raises(RuntimeError, match="Cannot load pretrained CAMPPlus") as caught:
        tts._init_pytorch_modules()
    assert isinstance(caught.value.__cause__, OSError)
    assert tts.campplus is None


def test_release_reference_encoders_preserves_conditioning_and_allows_reload():
    from mlx_indextts.generate_v25 import IndexTTSv25

    tts = IndexTTSv25.__new__(IndexTTSv25)
    for name in ("semantic_model", "campplus", "semantic_mean", "semantic_std", "extract_features"):
        setattr(tts, name, object())
    tts._preprocessing_initialized = True
    cached = {"speaker.wav": {"style": object()}}
    tts._reference_cache = cached
    tts.release_preprocessing_models()
    assert not tts._preprocessing_initialized
    assert tts.semantic_model is None and tts.campplus is None
    assert tts._reference_cache is cached
