"""Specialized inference kernels preserve filter phase, edges and parameter rounding."""

import mlx.core as mx
import numpy as np
import pytest
from mlx_indextts.models.activations import Activation1d, SnakeBeta, Snake


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 2), (1, 7, 7), (2, 8, 17), (1, 24, 257)])
@pytest.mark.parametrize("parameter_dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_metal_matches_reference(shape, parameter_dtype, noncontiguous, monkeypatch):
    from mlx_indextts.models import alias_free_metal

    original = alias_free_metal.snakebeta_alias_metal
    calls = []

    def tracked(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(alias_free_metal, "snakebeta_alias_metal", tracked)
    rng = np.random.default_rng(44)
    model = Activation1d(SnakeBeta(shape[1]))
    model.act.alpha = mx.array(rng.uniform(-2, 2, shape[1]).astype(np.float32)).astype(parameter_dtype)
    model.act.beta = mx.array(rng.uniform(-2, 2, shape[1]).astype(np.float32)).astype(parameter_dtype)
    raw = mx.array(rng.normal(size=shape).astype(np.float32))
    x = raw
    # Force a genuinely strided view with the same intended shape.
    if noncontiguous:
        x = mx.concatenate([raw, raw], axis=2)[:, :, ::2]
    expected = np.array(model(x))
    model.backend = "metal"
    actual = np.array(model(x))
    assert calls == [True]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    # Updated parameters must not use stale compiled constants.
    model.act.alpha = model.act.alpha + 0.1
    actual = np.array(model(x))
    model.backend = "reference"
    np.testing.assert_allclose(actual, np.array(model(x)), rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("kind", ["snake", "ratio3", "kernel8", "fp16", "linear_scale"])
def test_unsupported_configuration_falls_back_exactly(kind):
    act = Snake(3) if kind == "snake" else SnakeBeta(3, alpha_logscale=kind != "linear_scale")
    model = Activation1d(
        act,
        up_ratio=3 if kind == "ratio3" else 2,
        down_ratio=3 if kind == "ratio3" else 2,
        up_kernel_size=8 if kind == "kernel8" else 12,
    )
    x = mx.ones((1, 3, 17), dtype=mx.float16 if kind == "fp16" else mx.float32)
    expected = np.array(model(x))
    model.backend = "metal"
    np.testing.assert_array_equal(np.array(model(x)), expected)


def test_asymmetric_filters_preserve_convolution_orientation():
    """Symmetric Kaiser taps alone cannot reveal an accidental kernel reversal."""
    model = Activation1d(SnakeBeta(3))
    model.upsample._filter = mx.arange(1, 13, dtype=mx.float32).reshape(1, 12, 1) / 78
    model.downsample._filter = mx.arange(12, 0, -1, dtype=mx.float32).reshape(1, 12, 1) / 78
    x = mx.arange(51, dtype=mx.float32).reshape(1, 3, 17) / 51
    expected = np.array(model(x))
    model.backend = "metal"
    np.testing.assert_allclose(np.array(model(x)), expected, rtol=3e-6, atol=3e-6)
