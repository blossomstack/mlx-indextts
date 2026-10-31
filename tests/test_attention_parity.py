"""Independent PyTorch CPU oracle for MLX attention, including cached decoding.

Uses deterministic random tensors, not pretrained weights. This is operator
parity, not evidence of official IndexTTS checkpoint or speech-quality parity.
"""

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mlx_indextts.models.gpt2 import GPT2Attention, GPT2Model


@pytest.mark.parametrize("length,cached", [(1, 0), (17, 0), (1, 19), (5, 19)])
@pytest.mark.parametrize("dtype,atol", [(mx.float32, 2e-5), (mx.float16, 2e-3)])
def test_fused_attention_matches_torch_cpu(length, cached, dtype, atol):
    rng = np.random.default_rng(2026)
    model = GPT2Attention(dim=32, num_heads=4)
    for layer in (model.c_attn, model.c_proj):
        layer.weight = mx.array(rng.normal(0, 0.1, layer.weight.shape), dtype=dtype)
        layer.bias = mx.array(rng.normal(0, 0.1, layer.bias.shape), dtype=dtype)
    x = mx.array(rng.normal(size=(2, length, 32)), dtype=dtype)
    cache = tuple(mx.array(rng.normal(size=(2, cached, 32)), dtype=dtype)
                  for _ in range(2)) if cached else None
    positions = mx.arange(cached + length)[None, :] > (cached + mx.arange(length)[:, None])
    mask = mx.where(positions, -1e9, 0.0)[None, None]
    # Exercise left padding as well as causality. Keep at least one valid key.
    mask = mx.broadcast_to(mask, (2, 1, length, cached + length))
    if cached:
        mask[1, :, :, :2] = -1e9

    def tensor(value):
        return torch.from_numpy(np.array(value, dtype=np.float32))

    q, k, v = F.linear(tensor(x), tensor(model.c_attn.weight),
                        tensor(model.c_attn.bias)).chunk(3, dim=-1)
    if cache:
        k = torch.cat((tensor(cache[0]), k), dim=1)
        v = torch.cat((tensor(cache[1]), v), dim=1)
    expected_cache = (k, v)
    q, k, v = [a.reshape(2, -1, 4, 8).transpose(1, 2) for a in (q, k, v)]
    scores = q @ k.transpose(-1, -2) / np.sqrt(8) + tensor(mask)
    result = (scores.softmax(-1) @ v).transpose(1, 2).reshape(2, length, 32)
    expected = F.linear(result, tensor(model.c_proj.weight), tensor(model.c_proj.bias))
    actual, actual_cache = model(x, mask=mask, cache=cache)
    mx.eval(actual, actual_cache)
    np.testing.assert_allclose(np.array(actual), expected.numpy(), atol=atol, rtol=atol)
    for actual_kv, expected_kv in zip(actual_cache, expected_cache):
        np.testing.assert_allclose(np.array(actual_kv), expected_kv.numpy(), atol=atol, rtol=atol)


def test_incremental_backbone_matches_full_prefill():
    mx.random.seed(19)
    model = GPT2Model(dim=32, num_heads=4, num_layers=2)
    x = mx.random.normal((2, 11, 32))
    full, _ = model(x)
    prefix, cache = model(x[:, :7])
    chunks = [prefix]
    for index in range(7, 11):
        step, cache = model(x[:, index:index + 1], cache=cache)
        chunks.append(step)
    incremental = mx.concatenate(chunks, axis=1)
    mx.eval(full, incremental)
    np.testing.assert_allclose(np.array(full), np.array(incremental), atol=2e-5, rtol=2e-5)


def test_fully_left_padded_fp16_row_stays_finite():
    model = GPT2Attention(dim=32, num_heads=4)
    model.set_dtype(mx.float16)
    output, _ = model(mx.ones((1, 4, 32), dtype=mx.float16),
                      mask=mx.full((1, 1, 4, 4), -1e9))
    mx.eval(output)
    assert bool(mx.isfinite(output).all())
