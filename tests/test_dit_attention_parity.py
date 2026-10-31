"""Independent Torch oracle for fused DiT attention and grouped QKV slicing."""
import numpy as np
import mlx.core as mx
import pytest
import torch
import torch.nn.functional as F
from mlx_indextts.models.s2mel.dit import Attention,ModelArgs

@pytest.mark.parametrize('local_heads',[4,2])
@pytest.mark.parametrize('dtype,atol',[(mx.float32,2e-5),(mx.float16,0.003)])
def test_dit_attention_matches_torch_with_padding(local_heads,dtype,atol):
    rng=np.random.default_rng(81);b,t,h,d=2,17,4,8
    model=Attention(ModelArgs(dim=h*d,n_head=h,n_local_heads=local_heads))
    model.wqkv.weight=mx.array(rng.normal(0,.1,model.wqkv.weight.shape),dtype=dtype)
    model.wo.weight=mx.array(rng.normal(0,.1,model.wo.weight.shape),dtype=dtype)
    x=mx.array(rng.normal(size=(b,t,h*d)),dtype=dtype)
    angles=np.arange(t)[:,None]*np.exp(-np.arange(d//2)/3)[None,:]
    freq=mx.array(np.stack([np.cos(angles),np.sin(angles)],axis=-1).astype(np.float32))
    mask=mx.where(mx.arange(t)[None,None,None,:]<mx.array([17,12])[:,None,None,None],0.,float('-inf'))
    td=torch.float16 if dtype==mx.float16 else torch.float32
    qkv=F.linear(torch.tensor(np.array(x),dtype=td),torch.tensor(np.array(model.wqkv.weight),dtype=td))
    q,k,v=torch.split(qkv,[h*d,local_heads*d,local_heads*d],dim=-1)
    def rotate(y,heads):
        y=y.reshape(b,t,heads,d//2,2).float();f=torch.tensor(np.array(freq))
        a=y[...,0]*f[None,:,None,:,0]-y[...,1]*f[None,:,None,:,1]
        z=y[...,1]*f[None,:,None,:,0]+y[...,0]*f[None,:,None,:,1]
        return torch.stack([a,z],-1).reshape(b,t,heads,d).transpose(1,2)
    q=rotate(q,h);k=rotate(k,local_heads).repeat_interleave(h//local_heads,1)
    v=v.reshape(b,t,local_heads,d).transpose(1,2).float().repeat_interleave(h//local_heads,1)
    prob=(q@k.transpose(-1,-2)*d**-.5+torch.tensor(np.array(mask))).softmax(-1)
    expected=F.linear((prob@v).transpose(1,2).reshape(b,t,h*d),torch.tensor(np.array(model.wo.weight)).float()).numpy()
    actual=np.array(model(x,freq,mask))
    np.testing.assert_allclose(actual,expected,atol=atol,rtol=atol)
