"""Run actual manifest/version contract tests without importing the Apple runtime.

A namespace package exposes the pure-Python modules, leaving their implementations
unchanged. GPU/runtime tests remain separate and require Apple Silicon/Metal.
"""

import sys
import types
from pathlib import Path
import pytest

root = Path(__file__).resolve().parents[1]
package = types.ModuleType("mlx_indextts")
package.__path__ = [str(root / "mlx_indextts")]
sys.modules["mlx_indextts"] = package
raise SystemExit(
    pytest.main(
        [
            "--noconftest",
            "-q",
            str(root / "tests/test_model_manifest.py"),
            str(root / "tests/test_model_version.py"),
        ]
    )
)
