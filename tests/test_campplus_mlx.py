"""Targeted parity around CAMPPlus pooling boundaries and layout conversion."""
import numpy as np
import mlx.core as mx
import torch
import torch.nn.functional as F
import pytest
from mlx_indextts.models.campplus import CAMPPlus, convert_state_dict


@pytest.mark.parametrize('frames', [1, 99, 100, 101, 199, 200, 201])
def test_partial_segment_pool_excludes_padding(frames):
    x = np.random.default_rng(frames).normal(size=(2, frames, 7)).astype(np.float32)
    actual = np.array(CAMPPlus.segment_pool(mx.array(x)))
    reference = F.avg_pool1d(torch.from_numpy(x).transpose(1,2), 100, 100, ceil_mode=True)
    reference = reference.repeat_interleave(100, -1)[:,:,:frames].transpose(1,2).numpy()
    np.testing.assert_allclose(actual, reference, atol=1e-6)


def test_checkpoint_convolution_layout():
    state = {'conv.weight': torch.arange(24).reshape(2,3,4).float(), 'bn.num_batches_tracked': torch.tensor(4)}
    converted = convert_state_dict(state)
    assert set(converted) == {'conv.weight'}
    np.testing.assert_array_equal(np.array(converted['conv.weight']), state['conv.weight'].numpy().transpose(0,2,1))
