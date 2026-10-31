"""Native reference correctness: masking, causal convolution and DSP edge cases."""
import numpy as np
import mlx.core as mx
import pytest
import torch
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel
from mlx_indextts.models.w2v_bert import W2VBert
from mlx_indextts.audio_native import seamless_features,resample


def small_models():
    torch.manual_seed(7)
    cfg=Wav2Vec2BertConfig(hidden_size=16,num_attention_heads=4,intermediate_size=24,
        feature_projection_input_dim=8,num_hidden_layers=2,conv_depthwise_kernel_size=5,
        position_embeddings_type='relative_key',left_max_position_embeddings=3,right_max_position_embeddings=1,
        apply_spec_augment=False,hidden_dropout=0,attention_dropout=0)
    reference=Wav2Vec2BertModel(cfg).eval()
    native=W2VBert.__new__(W2VBert);native.config=cfg.to_dict();native.output_layer=2
    native.weights={k:mx.array(v.detach().numpy().transpose(0,2,1) if v.ndim==3 else v.detach().numpy()) for k,v in reference.state_dict().items()}
    return reference,native


def test_relative_attention_padding_and_layer_outputs():
    ref,native=small_models()
    x=np.random.default_rng(4).normal(size=(2,9,8)).astype(np.float32)
    mask=np.array([[1]*9,[1]*6+[0]*3])
    with torch.inference_mode():expected=ref(torch.from_numpy(x),attention_mask=torch.from_numpy(mask),output_hidden_states=True)
    _,stages=native(mx.array(x),mx.array(mask).astype(mx.bool_),return_stages=True)
    for actual,baseline in zip(stages,expected.hidden_states):
        np.testing.assert_allclose(np.array(actual),baseline.numpy(),atol=3e-6,rtol=2e-5)


def test_conformer_convolution_is_left_padded():
    _,native=small_models()
    x=mx.array(np.random.default_rng(2).normal(size=(1,10,16)).astype(np.float32))
    full=native.convolution(x,'encoder.layers.0.conv_module')
    prefix=native.convolution(x[:,:6],'encoder.layers.0.conv_module')
    np.testing.assert_allclose(np.array(full[:,:6]),np.array(prefix),atol=1e-6)


def test_silent_features_are_exact_zero_and_odd_pair_masked():
    # 15,900 samples gives 97 frames: final stacked pair contains padding.
    features,mask=seamless_features(mx.zeros(15900))
    assert features.shape==(1,49,160)
    np.testing.assert_array_equal(np.array(features[0,:-1]),0)
    assert not bool(mask[0,-1].item())
    np.testing.assert_array_equal(np.array(features[0,-1,80:]),1)


def test_reference_short_audio_rejected():
    with pytest.raises(ValueError,match='at least'):
        seamless_features(mx.zeros(100))


def test_resampling_length_and_identity():
    x=mx.array(np.arange(1000,dtype=np.float32)/1000)
    np.testing.assert_array_equal(np.array(resample(x,16000,16000)),np.array(x))
    assert resample(x,44100,22050).shape==(500,)
