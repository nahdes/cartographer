"""
tests/conftest.py  — pytest configuration + minimal shim for stdlib unittest.
When pytest is not installed, this provides a stub so tests can also be
run with:  python3 -m unittest discover -s tests
"""
try:
    import pytest  # noqa: F401 — pytest is the primary runner
except ImportError:
    # Minimal stub so test files don't crash on import when pytest is absent
    import sys
    import types
    import unittest

    _pytest_mod = types.ModuleType("pytest")

    def fixture(func=None, *, scope="function", autouse=False, params=None):
        """No-op decorator — unittest handles setup via setUp()."""
        if func is None:
            return lambda f: f
        return func

    def raises(exc, *args, **kwargs):
        return unittest.TestCase().assertRaises(exc)

    _pytest_mod.fixture = fixture
    _pytest_mod.raises  = raises
    _pytest_mod.mark    = types.SimpleNamespace(
        parametrize=lambda *a, **k: (lambda f: f),
        skip=lambda *a, **k: (lambda f: f),
        skipif=lambda *a, **k: (lambda f: f),
    )
    sys.modules["pytest"] = _pytest_mod
