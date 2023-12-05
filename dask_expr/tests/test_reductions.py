from __future__ import annotations

from dask_expr import from_pandas
from dask_expr.tests._util import _backend_library

# Set DataFrame backend for this module
lib = _backend_library()


def test_monotonic_increasing():
    pdf1 = lib.DataFrame(
        {
            "a": range(20),
            "b": list(range(20))[::-1],
            "c": [0] * 20,
            "d": [0] * 5 + [1] * 5 + [0] * 10,
        }
    )
    df1 = from_pandas(pdf1, 4)

    assert df1["a"].is_monotonic_increasing().compute()
    assert not df1["b"].is_monotonic_increasing().compute()
    assert df1["c"].is_monotonic_increasing().compute()
    assert not df1["d"].is_monotonic_increasing().compute()


def test_monotonic_decreasing():
    pdf1 = lib.DataFrame(
        {
            "a": range(20),
            "b": list(range(20))[::-1],
            "c": [0] * 20,
            "d": [0] * 5 + [1] * 5 + [0] * 10,
        }
    )
    df1 = from_pandas(pdf1, 4)

    assert not df1["a"].is_monotonic_decreasing().compute()
    assert df1["b"].is_monotonic_decreasing().compute()
    assert df1["c"].is_monotonic_decreasing().compute()
    assert not df1["d"].is_monotonic_decreasing().compute()
