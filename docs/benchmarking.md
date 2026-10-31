# Reproducible local benchmarks

Use your own authorized reference WAV and text files. Keep those inputs, output audio, voice caches and raw reports private unless you have permission to share them. The repository includes no internal transcripts, recordings or raw machine evidence.

```sh
HF_HUB_OFFLINE=1 .venv-native/bin/python scripts/benchmark_native.py \
  --model-dir models/mlx-v25-8bit --reference reference.wav \
  --text-file mandarin.txt --language zh --backend reference --output outputs/zh-reference
HF_HUB_OFFLINE=1 .venv-native/bin/python scripts/benchmark_native.py \
  --model-dir models/mlx-v25-8bit --reference reference.wav \
  --text-file mandarin.txt --language zh --backend metal --output outputs/zh-metal
```

Repeat with English and both short and longer texts. Compare the same inputs, checkpoints, settings and seed. The script excludes the first synthesis as warmup, measures three trials, records actual audio duration/RTF/peak MLX allocation, checks finite output and inserted silence, and saves a float waveform for numerical comparison. Raw-reference setup is measured separately; synthesis reuses a speaker cache. Output directories must be new. No models are downloaded by the script.

For a stronger comparison, alternate run order and repeat fresh processes. Record hardware, operating system, MLX version, asset revisions, cache limits, thermal state and other workloads. MLX allocation is distinct from RSS; its memory limit is soft. First-call latency includes compilation/driver/cache effects, and OS caches are not reset by restarting Python. Avoid treating one trial or a synthetic activation microbenchmark as an end-to-end result.

## Observed development result

On an M2 Pro with 16 GB, MLX 0.31.1, GPT 8-bit/group-64 and other synthesis weights fp16, the native Metal anti-aliasing path reduced median inference time from 26.403 to 16.074 seconds for 30.245 seconds of Mandarin audio, and from 22.442 to 12.952 seconds for 24.434 seconds of English audio. Three measured trials followed one warmup; model workloads were sequential. Stage timers used evaluation barriers in both phases. Longer Mandarin used surprised emotion, English happy, alpha 0.6; the generic public script uses calm. The private inputs are not distributed, so this table is a historical result rather than a directly reproducible public fixture claim.

The vocoder stage fell from 12.950 to 2.524 seconds for Mandarin and 10.551 to 2.041 seconds for English. English's untouched stages also ran faster; attributing all 42% wall-time reduction to the kernel would overstate the controlled gain. Holding other stages fixed predicts about 38%. Plain MLX compilation improved activation microbenchmarks only about 2% and was not retained. Partial fusion combines 12-tap polyphase interpolation with SnakeBeta, then performs the unchanged 12-tap decimation in a second kernel. The 2× intermediate remains. No quality setting or arithmetic precision was lowered.

Semantic tokens, durations, inserted 250 ms pauses and reference features were identical. Waveform RMS differences were 1.0–2.4e-6, maximum error 3.70e-4. Clipped-sample fractions were unchanged. Same-model ASR transcripts were identical, but this neither proves subjective equivalence nor fixes known wording/transcription discrepancies. Human listening and careful alignment remain necessary quality checks.

Hosted CI checks portable metadata contracts, critical lint, syntax and distributions. Native numerical tests and real-weight audio validation require a Metal-capable Apple Silicon machine and separately obtained assets. Do not report hosted CI success as speech-quality or GPU correctness validation.
