"""Import boundary and fallback behavior, independent of model weights."""
import subprocess
import sys
from pathlib import Path
import pytest


def test_runtime_imports_do_not_attempt_torch_or_transformers():
    result=subprocess.run([sys.executable,'-c', '''
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','torchaudio','transformers'}:
            raise AssertionError('unexpected import '+fullname)
sys.meta_path.insert(0,Block())
from mlx_indextts import IndexTTS, IndexTTSv2, IndexTTSv25
assert not any(n.split('.')[0] in {'torch','torchaudio','transformers'} for n in sys.modules)
'''],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


def test_missing_reference_dependency_has_actionable_error(monkeypatch):
    import mlx_indextts.optional_torch as optional
    def unavailable(name):
        raise ModuleNotFoundError(name,name=name)
    monkeypatch.setattr(optional,'import_module',unavailable)
    with pytest.raises(ModuleNotFoundError,match=r'mlx-indextts\[reference\]'):
        optional.OptionalModule('torch').load


def test_deferred_no_grad_preserves_reference_semantics():
    import torch
    from mlx_indextts.optional_torch import torch_no_grad
    @torch_no_grad
    def compute(x):
        return x*2
    x=torch.tensor([1.0],requires_grad=True)
    assert not compute(x).requires_grad
    assert torch.is_grad_enabled()
