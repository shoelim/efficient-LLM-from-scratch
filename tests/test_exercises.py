"""Verify the included first reference solution, not the deliberately blank exercise."""

import importlib.util
from pathlib import Path

import pytest


def _load_solution():
    path = Path(__file__).resolve().parents[1] / "exercises/solutions/00_python_basics.py"
    spec = importlib.util.spec_from_file_location("basics_solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_first_solution():
    module = _load_solution()
    assert module.count_tokens(["a", "b", "a"]) == {"a": 2, "b": 1}
    assert module.count_tokens([]) == {}
    assert module.make_lm_examples([1, 2, 3, 4], 2) == [([1, 2], 3), ([2, 3], 4)]
    assert module.make_lm_examples([1], 2) == []
    with pytest.raises(ValueError):
        module.make_lm_examples([1, 2], 0)
