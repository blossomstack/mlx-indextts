"""Long fp16 normalization must agree with a PyTorch CPU fp32 oracle."""

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mlx_indextts.models.s2mel.length_regulator import GroupNorm


@pytest.mark.parametrize("groups,channels,length", [(1, 512, 1069), (4, 32, 1024), (1, 8, 11)])
@pytest.mark.parametrize("dtype,atol", [(mx.float16, 0.003), (mx.float32, 2e-5)])
def test_groupnorm_long_reduction_matches_torch(groups, channels, length, dtype, atol):
    rng = np.random.default_rng(27)
    x = mx.array(rng.normal(2.0, 0.4, (1, channels, length)), dtype=dtype)
    model = GroupNorm(groups, channels)
    model.weight = mx.array(rng.uniform(0.2, 1.3, channels), dtype=dtype)
    model.bias = mx.array(rng.normal(0, 0.1, channels), dtype=dtype)

    def tensor(a):
        return torch.from_numpy(np.array(a, dtype=np.float32))

    expected = F.group_norm(tensor(x), groups, tensor(model.weight), tensor(model.bias), eps=model.eps)
    actual = model(x)
    mx.eval(actual)
    assert actual.dtype == dtype
    assert bool(mx.isfinite(actual).all())
    np.testing.assert_allclose(np.array(actual), expected.numpy(), atol=atol, rtol=atol)


def test_nonfinite_stage_raises_instead_of_writing_broken_audio():
    from mlx_indextts.generate_v25 import _require_finite

    _require_finite(mx.array([1.0, 0.0]), "test")
    with pytest.raises(FloatingPointError, match="length regulator"):
        _require_finite(mx.array([float("nan")]), "length regulator")
