"""Torch-free MLX reference audio DSP; NumPy only builds fixed coefficients.

Kaldi-compatible fbank and sinc/Hann resampling follow torchaudio (BSD-2-Clause).
Seamless feature stacking follows Transformers (Apache-2.0). WAV decoding is
handled separately by soundfile. All signal transforms below execute in MLX.
"""
import math
import numpy as np
import mlx.core as mx


def resample(audio,source_rate,target_rate):
    if source_rate==target_rate:return audio
    if source_rate<=0 or target_rate<=0:raise ValueError('Sample rates must be positive')
    gcd=math.gcd(source_rate,target_rate);old=source_rate//gcd;new=target_rate//gcd
    base=min(old,new)*0.99;width=math.ceil(6*old/base)
    idx=mx.arange(-width,width+old,dtype=mx.float32)[None,:]/old
    t=(mx.arange(0,-new,-1,dtype=mx.float32)[:,None]/new+idx)*base
    t=mx.clip(t,-6,6)
    window=mx.cos(t*math.pi/6/2)**2;t=t*math.pi
    kernel=mx.where(t==0,1,mx.sin(t)/mx.where(t==0,1,t))*window*(base/old)
    x=mx.pad(audio,[(width,width+old)])
    y=mx.conv1d(x[None,:,None],kernel[:,:,None],stride=old)
    return y.reshape(-1)[:math.ceil(new*audio.shape[0]/old)]


def frames(audio,size,hop):
    if audio.ndim!=1 or len(audio)<size:raise ValueError(f'Audio must be mono with at least {size} samples')
    indices=mx.arange((len(audio)-size)//hop+1)[:,None]*hop+mx.arange(size)[None,:]
    return audio[indices]


def kaldi_filters():
    # Kaldi triangular filters are linear in mel, with no area normalization.
    mel=lambda f:1127*np.log1p(f/700)
    points=np.linspace(mel(20),mel(8000),82)
    freq=mel(np.arange(257)*16000/512)
    lower=(freq[:,None]-points[:-2])/(points[1:-1]-points[:-2])
    upper=(points[2:]-freq[:,None])/(points[2:]-points[1:-1])
    return mx.array(np.maximum(0,np.minimum(lower,upper)).astype(np.float32))


def fbank(audio):
    x=frames(audio,400,160)
    x=x-mx.mean(x,axis=1,keepdims=True)
    previous=mx.concatenate((x[:,:1],x[:,:-1]),axis=1)
    x=x-0.97*previous
    window=mx.array((np.hanning(400)**0.85).astype(np.float32))
    spec=mx.fft.rfft(x*window,n=512,axis=-1)
    power=spec.real**2+spec.imag**2
    return mx.log(mx.maximum(power @ kaldi_filters(),np.finfo(np.float32).eps))


def seamless_features(audio):
    x=fbank(audio*32768)
    count=x.shape[0]
    if count<2:raise ValueError('Reference needs at least two fbank frames')
    # Shift before reduction so a constant/silent channel normalizes to zero
    # rather than amplifying fp32 reduction roundoff by rsqrt(1e-7).
    x=x-x[:1]
    x=x-mx.mean(x,axis=0,keepdims=True)
    x=x*mx.rsqrt(mx.mean(x*x,axis=0,keepdims=True)*(count/(count-1))+1e-7)
    if count%2:x=mx.pad(x,[(0,1),(0,0)],constant_values=1)
    mask=mx.arange(x.shape[0])<count
    return x.reshape(1,-1,160),mask[1::2][None,:]


def mel_spectrogram(audio,*,sr=22050,n_fft=1024,hop_length=256,win_length=1024,n_mels=80,fmin=0,fmax=None):
    from librosa.filters import mel
    pad=(n_fft-hop_length)//2
    if len(audio)<=pad:raise ValueError('Reference too short for reflect padding')
    x=mx.concatenate((audio[1:pad+1][::-1],audio,audio[-pad-1:-1][::-1]))
    x=frames(x,n_fft,hop_length)
    window=0.5-0.5*mx.cos(2*math.pi*mx.arange(win_length)/win_length)
    left=(n_fft-win_length)//2
    window=mx.pad(window,[(left,n_fft-win_length-left)])
    spec=mx.fft.rfft(x*window,axis=-1)
    mag=mx.sqrt(spec.real**2+spec.imag**2+1e-9)
    basis=mx.array(mel(sr=sr,n_fft=n_fft,n_mels=n_mels,fmin=fmin,fmax=fmax))
    return mx.log(mx.maximum(mag @ basis.T,1e-5)).T[None,:,:]
