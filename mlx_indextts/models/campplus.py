# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Adapted for MLX under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
"""Frozen fp32 MLX CAMPPlus speaker encoder (80-bin fbank -> 192-vector).

Architecture derived from 3D-Speaker CAMPPlus, Apache-2.0:
https://github.com/alibaba-damo-academy/3D-Speaker
See indextts/s2mel/modules/campplus for the attributed PyTorch reference.
No Torch operations are used by this module. Input features remain a separate stage.
"""
import mlx.core as mx


class CAMPPlus:
    def __init__(self, weights):
        self.weights = weights

    @classmethod
    def load(cls, path):
        return cls(mx.load(str(path)))

    def _conv(self, x, key, stride=1, padding=0, dilation=1):
        w = self.weights[key + '.weight']
        if w.ndim == 4:
            y = mx.conv2d(x, w, stride=stride, padding=padding)
        else:
            y = mx.conv1d(x, w, stride=stride, padding=padding, dilation=dilation)
        bias = self.weights.get(key + '.bias')
        return y if bias is None else y + bias

    def _bn(self, x, key):
        w = self.weights
        y = (x - w[key + '.running_mean']) * mx.rsqrt(w[key + '.running_var'] + 1e-5)
        if key + '.weight' in w:
            y = y * w[key + '.weight'] + w[key + '.bias']
        return y

    def _nonlinear(self, x, key):
        return mx.maximum(self._bn(x, key + '.batchnorm'), 0)

    @staticmethod
    def segment_pool(x, length=100):
        # The last partial segment divides by its actual length (ceil_mode).
        return mx.concatenate([mx.broadcast_to(mx.mean(x[:, i:i+length], axis=1, keepdims=True), x[:, i:i+length].shape)
                               for i in range(0, x.shape[1], length)], axis=1)

    def __call__(self, features, *, return_stages=False):
        if features.ndim != 3 or features.shape[-1] != 80 or features.shape[1] < 3:
            raise ValueError('CAMPPlus expects [batch, frames>=3, 80] fbank features')
        # NHWC: frequency, time, channel. Flatten channel before frequency,
        # exactly as the reference NCHW reshape does.
        x = features.astype(mx.float32).transpose(0, 2, 1)[:, :, :, None]
        x = mx.maximum(self._bn(self._conv(x, 'head.conv1', padding=1), 'head.bn1'), 0)
        for stage in (1, 2):
            for block in (0, 1):
                key = f'head.layer{stage}.{block}'
                stride = (2, 1) if block == 0 else (1, 1)
                residual = x
                y = mx.maximum(self._bn(self._conv(x, key+'.conv1', stride=stride, padding=1), key+'.bn1'), 0)
                y = self._bn(self._conv(y, key+'.conv2', padding=1), key+'.bn2')
                if block == 0:
                    residual = self._bn(self._conv(x, key+'.shortcut.0', stride=stride), key+'.shortcut.1')
                x = mx.maximum(y + residual, 0)
        x = mx.maximum(self._bn(self._conv(x, 'head.conv2', stride=(2,1), padding=1), 'head.bn2'), 0)
        x = x.transpose(0, 2, 3, 1).reshape(x.shape[0], x.shape[2], -1)
        stages = {'head': x}
        x = self._nonlinear(self._conv(x, 'xvector.tdnn.linear', stride=2, padding=2), 'xvector.tdnn.nonlinear')
        for block, (count, dilation) in enumerate(((12,1),(24,2),(16,2)), 1):
            for layer in range(1, count+1):
                key = f'xvector.block{block}.tdnnd{layer}'
                z = self._conv(self._nonlinear(x, key+'.nonlinear1'), key+'.linear1')
                z = self._nonlinear(z, key+'.nonlinear2')
                context = mx.mean(z, axis=1, keepdims=True) + self.segment_pool(z)
                context = mx.maximum(self._conv(context, key+'.cam_layer.linear1'), 0)
                mask = mx.sigmoid(self._conv(context, key+'.cam_layer.linear2'))
                local = self._conv(z, key+'.cam_layer.linear_local', padding=dilation, dilation=dilation)
                x = mx.concatenate((x, local * mask), axis=-1)
            key = f'xvector.transit{block}'
            x = self._conv(self._nonlinear(x, key+'.nonlinear'), key+'.linear')
            stages[f'transit{block}'] = x
        x = self._nonlinear(x, 'xvector.out_nonlinear')
        # Torch std defaults to Bessel correction (unbiased=True).
        std = mx.sqrt(mx.var(x, axis=1) * (x.shape[1] / (x.shape[1]-1)))
        pooled = mx.concatenate((mx.mean(x, axis=1), std), axis=-1)
        x = self._conv(pooled[:,None,:], 'xvector.dense.linear')[:,0,:]
        x = self._bn(x, 'xvector.dense.nonlinear.batchnorm')
        return (x, stages) if return_stages else x


def convert_state_dict(state):
    """Convert a CPU Torch state dict once; checkpoint is not needed at runtime."""
    result = {}
    for key, tensor in state.items():
        if key.endswith('num_batches_tracked'):
            continue
        array = tensor.detach().float().cpu().numpy()
        if array.ndim == 4:
            array = array.transpose(0,2,3,1)
        elif array.ndim == 3:
            array = array.transpose(0,2,1)
        result[key] = mx.array(array)
    return result
