"""Inference-only fp32 ratio-2/12-tap SnakeBeta anti-alias kernels.

Explicit opt-in from Activation1d; other configurations use the reference path.
Filters and parameters remain runtime inputs, so weight updates are respected.
"""

from functools import lru_cache

import mlx.core as mx

_UP_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    uint length = x_shape[2];
    uint channels = x_shape[1];
    uint t = i % (2 * length);
    uint bc = i / (2 * length);
    float v = 0.0f;
    for (int k = int((t + 1) % 2); k < 12; k += 2) {
        int j = clamp((int(t) + 5 - k) / 2, 0, int(length) - 1);
        v += x[bc * length + j] * uf[k];
    }
    v *= 2.0f;
    float a = alpha[bc % channels];
    float inv_beta = beta[bc % channels];
    float s = metal::sin(a * v);
    out[i] = v + inv_beta * (s * s);
"""
_DOWN_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    uint length = x_shape[2];
    uint t = i % (length / 2);
    uint bc = i / (length / 2);
    float v = 0.0f;
    for (int k = 0; k < 12; ++k) {
        int j = clamp(2 * int(t) - 5 + k, 0, int(length) - 1);
        v += x[bc * length + j] * df[k];
    }
    out[i] = v;
"""


@lru_cache(maxsize=1)
def _kernels():
    up = mx.fast.metal_kernel(
        name="alias_up_snake_fp32", input_names=["x", "alpha", "beta", "uf"], output_names=["out"], source=_UP_SOURCE
    )
    down = mx.fast.metal_kernel(
        name="alias_down_fp32", input_names=["x", "df"], output_names=["out"], source=_DOWN_SOURCE
    )
    return up, down


def snakebeta_alias_metal(x, alpha, beta, up_filter, down_filter):
    """Two kernels: fused polyphase interpolation + SnakeBeta, then FIR decimation.

    Edge padding and crop are folded into index arithmetic. The 2x intermediate
    remains; this is partial fusion, not a fully fused up/activation/down kernel.
    Parameter exp/reciprocal are evaluated in their original dtype before fp32
    multiplication, matching the reference even for fp16 checkpoint parameters.
    """
    up_kernel, down_kernel = _kernels()
    # Preserve parameter-dtype rounding of exp and reciprocal in the reference.
    alpha = mx.exp(alpha).astype(mx.float32)
    beta = (1.0 / (mx.exp(beta) + 1e-9)).astype(mx.float32)
    b, c, n = x.shape
    up = up_kernel(
        inputs=[x, alpha, beta, up_filter],
        grid=(b * c * n * 2, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, c, n * 2)],
        output_dtypes=[mx.float32],
    )[0]
    return down_kernel(
        inputs=[up, down_filter],
        grid=(b * c * n, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.float32],
    )[0]
