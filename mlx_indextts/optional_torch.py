"""Lazy compatibility boundary for optional PyTorch reference implementations.

Importing this module never imports Torch. Native MLX inference must never
access these proxies; reference/conversion users install the corresponding extra.
"""
from functools import wraps
from importlib import import_module


class OptionalModule:
    def __init__(self, name):
        self.name = name
        self._module = None

    def __getattr__(self, name):
        if self._module is None:
            try:
                self._module = import_module(self.name)
            except ModuleNotFoundError as exc:
                if exc.name == self.name:
                    raise ModuleNotFoundError(
                        f"The PyTorch reference backend requires {self.name}. "
                        "Install 'mlx-indextts[reference]' or select "
                        "reference_backend='mlx' with converted native assets."
                    ) from exc
                raise
        return getattr(self._module, name)


torch = OptionalModule('torch')
torchaudio = OptionalModule('torchaudio')


def torch_no_grad(function):
    """Enter Torch no_grad only when the reference method is called."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with torch.no_grad():
            return function(*args, **kwargs)
    return wrapped
