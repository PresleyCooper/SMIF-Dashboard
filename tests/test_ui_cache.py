"""
A guard on the UI cache keys.

Streamlit excludes any parameter whose name starts with an underscore from a
cache key — the documented way to pass an unhashable connection or client. The
UI's whole invalidation scheme is a fingerprint of (ledger mtime, price-cache
mtime) passed into every cached function, so naming that parameter `_fp` turns
every cache into a permanent one: the app would keep serving the previous
build's returns after a transaction is saved, and nothing would look broken.

It is a one-character mistake with no visible symptom, so it gets a test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

DATA_MODULE = Path(__file__).resolve().parent.parent / "ui" / "data.py"
CACHE_DECORATORS = {"cache_data", "cache_resource"}


def _cached_functions() -> list[ast.FunctionDef]:
    tree = ast.parse(DATA_MODULE.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            call = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(call, ast.Attribute) and call.attr in CACHE_DECORATORS:
                out.append(node)
                break
    return out


def test_the_ui_actually_has_cached_functions():
    """If this fails the rest of the file is vacuously passing."""
    assert len(_cached_functions()) >= 3


@pytest.mark.parametrize("fn", _cached_functions(),
                         ids=lambda f: f.name)
def test_no_cached_parameter_is_excluded_from_the_key(fn):
    args = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
    excluded = [a for a in args if a.startswith("_")]
    assert not excluded, (
        f"{fn.name}() takes {excluded}, which Streamlit drops from the cache "
        f"key — the result would be cached for ever and never refreshed")


@pytest.mark.parametrize("fn", _cached_functions(), ids=lambda f: f.name)
def test_every_cached_function_takes_the_fingerprint_first(fn):
    args = [a.arg for a in fn.args.args]
    assert args and args[0] == "fingerprint", (
        f"{fn.name}() must take the (ledger, price-cache) fingerprint as its "
        f"first argument so a saved transaction invalidates it")
