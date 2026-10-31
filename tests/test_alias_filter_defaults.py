"""Default anti-alias constructors must resolve the optional kernel size."""
import numpy as np
import mlx.core as mx
import pytest
from mlx_indextts.models.activations import UpSample1d,DownSample1d

@pytest.mark.parametrize('cls',[UpSample1d,DownSample1d])
@pytest.mark.parametrize('ratio',[2,3])
def test_default_kernel_matches_explicit_kernel(cls,ratio):
    x=mx.array(np.random.default_rng(4).normal(size=(1,3,24)).astype(np.float32))
    default=cls(ratio=ratio)
    explicit=cls(ratio=ratio,kernel_size=int(6*ratio//2)*2)
    np.testing.assert_array_equal(np.array(default(x)),np.array(explicit(x)))
