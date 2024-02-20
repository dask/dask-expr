from __future__ import annotations

import pytest

from dask_expr import from_pandas
from dask_expr.io import IO
from dask_expr.tests._util import _backend_library, assert_eq

# Set DataFrame backend for this module
pd = _backend_library()


@pytest.fixture
def pdf():
    pdf = pd.DataFrame({"x": range(100), "a": 1, "b": 1, "c": 1})
    pdf["y"] = pdf.x // 7  # Not unique; duplicates span different partitions
    yield pdf


@pytest.fixture
def df(pdf):
    yield from_pandas(pdf, npartitions=10)


def _check_io_nodes(expr, expected):
    expr = expr.optimize(fuse=False)
    io_nodes = list(expr.find_operations(IO))
    assert len(io_nodes) == expected
    assert len({node._branch_id.branch_id for node in io_nodes}) == expected


def test_reuse_everything_scalar_and_series(df, pdf):
    df["new"] = 1
    df["new2"] = df["x"] + 1
    df["new3"] = df.x[df.x > 1] + df.x[df.x > 2]

    pdf["new"] = 1
    pdf["new2"] = pdf["x"] + 1
    pdf["new3"] = pdf.x[pdf.x > 1] + pdf.x[pdf.x > 2]
    assert_eq(df, pdf)
    _check_io_nodes(df, 1)


def test_dont_reuse_reducer(df, pdf):
    result = df.replace(1, 5)
    result["new"] = result.x + result.y.sum()
    expected = pdf.replace(1, 5)
    expected["new"] = expected.x + expected.y.sum()
    assert_eq(result, expected)
    _check_io_nodes(result, 2)

    result = df + df.sum()
    expected = pdf + pdf.sum()
    assert_eq(result, expected, check_names=False)  # pandas 2.2 bug
    _check_io_nodes(result, 2)

    result = df.replace(1, 5)
    rhs_1 = result.x + result.y.sum()
    rhs_2 = result.b + result.a.sum()
    result["new"] = rhs_1
    result["new2"] = rhs_2
    expected = pdf.replace(1, 5)
    expected["new"] = expected.x + expected.y.sum()
    expected["new2"] = expected.b + expected.a.sum()
    assert_eq(result, expected)
    _check_io_nodes(result, 2)

    result = df.replace(1, 5)
    result["new"] = result.x + result.y.sum()
    result["new2"] = result.b + result.a.sum()
    expected = pdf.replace(1, 5)
    expected["new"] = expected.x + expected.y.sum()
    expected["new2"] = expected.b + expected.a.sum()
    assert_eq(result, expected)
    _check_io_nodes(result, 3)

    result = df.replace(1, 5)
    result["new"] = result.x + result.sum().dropna().prod()
    expected = pdf.replace(1, 5)
    expected["new"] = expected.x + expected.sum().dropna().prod()
    assert_eq(result, expected)
    _check_io_nodes(result, 2)
