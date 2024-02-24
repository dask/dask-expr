from __future__ import annotations

import pytest

from dask_expr import from_pandas
from dask_expr._merge import BlockwiseMerge
from dask_expr._shuffle import DiskShuffle
from dask_expr.tests._util import _backend_library, _check_consumer_node, assert_eq

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


def test_reuse_everything_scalar_and_series(df, pdf):
    df["new"] = 1
    df["new2"] = df["x"] + 1
    df["new3"] = df.x[df.x > 1] + df.x[df.x > 2]

    pdf["new"] = 1
    pdf["new2"] = pdf["x"] + 1
    pdf["new3"] = pdf.x[pdf.x > 1] + pdf.x[pdf.x > 2]
    assert_eq(df, pdf)
    _check_consumer_node(df, 1)


def test_dont_reuse_reducer(df, pdf):
    result = df.replace(1, 5)
    result["new"] = result.x + result.y.sum()
    expected = pdf.replace(1, 5)
    expected["new"] = expected.x + expected.y.sum()
    assert_eq(result, expected)
    _check_consumer_node(result, 2)

    result = df + df.sum()
    expected = pdf + pdf.sum()
    assert_eq(result, expected, check_names=False)  # pandas 2.2 bug
    _check_consumer_node(result, 2)

    result = df.replace(1, 5)
    rhs_1 = result.x + result.y.sum()
    rhs_2 = result.b + result.a.sum()
    result["new"] = rhs_1
    result["new2"] = rhs_2
    expected = pdf.replace(1, 5)
    expected["new"] = expected.x + expected.y.sum()
    expected["new2"] = expected.b + expected.a.sum()
    assert_eq(result, expected)
    _check_consumer_node(result, 2)

    result = df.replace(1, 5)
    result["new"] = result.x + result.y.sum()
    result["new2"] = result.b + result.a.sum()
    expected = pdf.replace(1, 5)
    expected["new"] = expected.x + expected.y.sum()
    expected["new2"] = expected.b + expected.a.sum()
    assert_eq(result, expected)
    _check_consumer_node(result, 3)

    result = df.replace(1, 5)
    result["new"] = result.x + result.sum().dropna().prod()
    expected = pdf.replace(1, 5)
    expected["new"] = expected.x + expected.sum().dropna().prod()
    assert_eq(result, expected)
    _check_consumer_node(result, 2)


def test_disk_shuffle(df, pdf):
    q = df.shuffle("a")
    q = q.fillna(100)
    q = q.a + q.a.sum()
    q.optimize(fuse=False).pprint()
    # Disk shuffle is not utilizing pipeline breakers
    _check_consumer_node(q, 1, consumer_node=DiskShuffle)
    _check_consumer_node(q, 1)
    expected = pdf.fillna(100)
    expected = expected.a + expected.a.sum()
    assert_eq(q, expected)


def test_groupb_apply_disk_shuffle_reuse(df, pdf):
    q = df.groupby("a").apply(lambda x: x)
    q = q.fillna(100)
    q = q.a + q.a.sum()
    # Disk shuffle is not utilizing pipeline breakers
    _check_consumer_node(q, 1, consumer_node=DiskShuffle)
    _check_consumer_node(q, 1)
    expected = pdf.groupby("a").apply(lambda x: x)
    expected = expected.fillna(100)
    expected = expected.a + expected.a.sum()
    assert_eq(q, expected)


def test_groupb_ffill_disk_shuffle_reuse(df, pdf):
    q = df.groupby("a").ffill()
    q = q.fillna(100)
    q = q.b + q.b.sum()
    # Disk shuffle is not utilizing pipeline breakers
    _check_consumer_node(q, 1, consumer_node=DiskShuffle)
    _check_consumer_node(q, 1)
    expected = pdf.groupby("a").ffill()
    expected = expected.fillna(100)
    expected = expected.b + expected.b.sum()
    assert_eq(q, expected)


def test_merge_reuse():
    pdf1 = pd.DataFrame({"a": [1, 2, 3, 4, 1, 2, 3, 4], "b": 1, "c": 1})
    pdf2 = pd.DataFrame({"a": [1, 2, 3, 4, 1, 2, 3, 4], "e": 1, "f": 1})

    df1 = from_pandas(pdf1, npartitions=3)
    df2 = from_pandas(pdf2, npartitions=3)
    q = df1.merge(df2)
    q = q.fillna(100)
    q = q.b + q.b.sum()
    _check_consumer_node(q, 1, BlockwiseMerge)
    # One on either side
    _check_consumer_node(q, 2, DiskShuffle, branch_id_counter=1)
    _check_consumer_node(q, 2, branch_id_counter=1)

    expected = pdf1.merge(pdf2)
    expected = expected.fillna(100)
    expected = expected.b + expected.b.sum()
    assert_eq(q, expected, check_index=False)
