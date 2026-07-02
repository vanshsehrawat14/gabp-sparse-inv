"""M-JOSS packaging guards: the public API is importable and self-consistent.

These catch the cheap regressions that break a "hot-swappable drop-in": a dangling
``__all__`` export, a missing ``__version__`` (the pyproject dynamic version reads
``gabp_sparse_inv.__version__``), or an accidental duplicate export.
"""

from __future__ import annotations

import collections

import gabp_sparse_inv


def test_version_is_nonempty_string():
    v = gabp_sparse_inv.__version__
    assert isinstance(v, str) and v                       # pyproject dynamic version reads this


def test_all_exports_resolve():
    missing = [name for name in gabp_sparse_inv.__all__ if not hasattr(gabp_sparse_inv, name)]
    assert not missing, f"names in __all__ not importable: {missing}"


def test_all_has_no_duplicates():
    dupes = [n for n, c in collections.Counter(gabp_sparse_inv.__all__).items() if c > 1]
    assert not dupes, f"duplicate names in __all__: {dupes}"
