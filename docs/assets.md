# Model assets and conversion

No download is performed by installing this package. Review [the model license](../licenses/LicenseRef-bilibili.txt), [NOTICE](../NOTICE.md) and each upstream model card first. The commands below obtain models only when you run them yourself. Converted weights retain the original terms; do not treat the source MIT license as permission to redistribute every asset.

Use a separate conversion environment on Apple Silicon:

```sh
python3.12 -m venv .venv-convert
.venv-convert/bin/python -m pip install '.[native,convert]'
.venv-convert/bin/hf download IndexTeam/IndexTTS-2.5 \
  --revision d0aa86e75bb6f3437f3831e95056fa72842d89ef \
  --local-dir models/IndexTTS-2.5-source
.venv-convert/bin/mlx-indextts convert \
  --model-dir models/IndexTTS-2.5-source --output models/mlx-v25-8bit \
  --dtype float16 --quantize 8 \
  --source-revision d0aa86e75bb6f3437f3831e95056fa72842d89ef
```

The converter may obtain its required upstream BigVGAN checkpoint if not cached. Its source is `nvidia/bigvgan_v2_22khz_80band_256x` at `633ff708ed5b74903e86ff1298cf4a98e921c513`, under MIT. GPT uses persistent 8-bit/group-64 quantization; the other synthesis modules retain fp16 weights. Conversion-time Torch dependencies are separate from native inference.

For native raw-WAV conditioning, obtain the original W2V-BERT config/weights and CAMPPlus checkpoint:

```sh
.venv-convert/bin/hf download facebook/w2v-bert-2.0 \
  --include config.json model.safetensors \
  --local-dir models/facebook-w2v-bert-2.0
.venv-convert/bin/hf download funasr/campplus campplus_cn_common.bin \
  --revision e4b6ede7ce16997aff4ae69fbca1f0175e2afede \
  --local-dir models/campplus-source
.venv-convert/bin/python scripts/convert_native_reference.py \
  --model-dir models/mlx-v25-8bit \
  --campplus-checkpoint models/campplus-source/campplus_cn_common.bin
```

W2V-BERT is read directly as safetensors. You may instead set `INDEXTTS_W2V_BERT_DIR` to the directory containing its `config.json` and `model.safetensors`. Pin and record the resolved upstream revision for reproducible deployments. CAMPPlus and reference/emotion tensors are converted to safetensors by the helper. It refuses to overwrite existing native reference assets.

The converted model directory must contain the synthesis manifest/config, GPT/codec/S2Mel/BigVGAN tensors, tokenizer, `campplus.safetensors` and `reference-stats.safetensors`. Keep all original license information with assets. The legacy `.pt` files are conversion/reference-backend inputs; native inference reads the converted tensors.

After conversion, return to the native-only environment and explicitly use `reference_backend="mlx"` as shown in the README. The optional reference path uses `.[native,reference]`. Do not install `dev`, `convert`, `reference` or the compatibility `v25` extra into an environment you intend to keep Torch-free.
