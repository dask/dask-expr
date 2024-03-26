import numpy as np
import pytest

from dask_expr import from_pandas, read_parquet
from dask_expr._shuffle import DiskShuffle, Shuffle
from dask_expr.tests._util import _backend_library, assert_eq

# Set DataFrame backend for this module
pd = _backend_library()


@pytest.mark.xfail(
    reason="propagate a different property because it's not sufficient for merges"
)
def test_groupby_implicit_divisions(tmpdir):
    pdf1 = pd.DataFrame({"a": range(10), "bb": 1})

    df1 = from_pandas(pdf1, npartitions=2)
    df1.to_parquet(tmpdir / "df1.parquet")
    df1 = read_parquet(
        tmpdir / "df1.parquet", filesystem="pyarrow", calculate_divisions=True
    )

    result = df1.groupby("a").apply(lambda x: x + 1).optimize()
    assert not list(result.find_operations(Shuffle))
    assert len(result.compute()) == 10


def test_groupby_avoid_shuffle():
    pdf1 = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6] * 100, "b": 1, "c": 2})
    pdf2 = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6] * 100, "d": 1, "e": 2})

    df1 = from_pandas(pdf1, npartitions=4)
    df2 = from_pandas(pdf2, npartitions=3)
    q = df1.merge(df2)
    q = q.groupby("a").sum(split_out=True)
    result = q.optimize(fuse=False)
    assert (
        len(list(node for node in result.walk() if isinstance(node, DiskShuffle))) == 2
    )

    expected = pdf1.merge(pdf2)
    expected = expected.groupby("a").sum()
    assert_eq(q, expected, check_index=False)


@pytest.mark.parametrize(
    "func",
    [
        lambda x: x.replace(1, 5),
        lambda x: x.fillna(100),
        lambda x: np.log(x),
        lambda x: x.combine_first(x),
        lambda x: x.query("a > 2"),
        lambda x: x.b.isin([1]),
        lambda x: x.eval("z=a+b"),
        lambda x: x + x,
    ],
)
def test_shuffle_when_necessary(func):
    pdf = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6] * 100, "b": 1, "c": 2})

    df = from_pandas(pdf, npartitions=4)
    q = df.groupby("a").sum(split_out=True).reset_index()
    q = func(q)
    assert q.unique_partition_mapping_columns == set()
    expected = pdf.groupby("a").sum().reset_index()
    expected = func(expected)
    assert_eq(q, expected, check_index=False)


@pytest.mark.parametrize(
    "func",
    [
        lambda x: x["a"],
        lambda x: x[x.a > 5],
        lambda x: x.drop(columns=["a"]),
        lambda x: x.drop_duplicates(),
        lambda x: x.dropna(),
        lambda x: x.rename(columns={"a": "x"}),
        lambda x: x.assign(z=x.a + x.b),
        lambda x: x.assign(b=x.a + x.b),
    ],
)
def test_avoid_shuffle_when_possible(func):
    pdf = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6] * 100, "b": 1, "c": 2})

    df = from_pandas(pdf, npartitions=4)
    q = df.groupby("a").sum(split_out=True).reset_index()
    q = func(q)
    assert q.unique_partition_mapping_columns == {("a",)}
    expected = pdf.groupby("a").sum().reset_index()
    expected = func(expected)
    assert_eq(q, expected, check_index=False)


def test_repartition():
    pdf = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6] * 100, "b": 1, "c": 2})

    df = from_pandas(pdf, npartitions=4)
    q = df.groupby("a").sum(split_out=True).reset_index()
    q = q.repartition(npartitions=3)
    assert q.unique_partition_mapping_columns == {("a",)}

    q = q.repartition(npartitions=5)
    assert q.unique_partition_mapping_columns == set()


def test_shuffle():
    pdf = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6] * 100, "b": 1, "c": 2})

    df = from_pandas(pdf, npartitions=4)
    q = df.shuffle("a")
    assert q.unique_partition_mapping_columns == {("a",)}


def test_merge_avoid_shuffle():
    pdf = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6] * 100, "b": 1, "c": 2})
    pdf2 = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6] * 100, "d": 1, "e": 2})

    df = from_pandas(pdf, npartitions=4)
    df2 = from_pandas(pdf2, npartitions=3)
    q = df.groupby("a").sum(split_out=True).reset_index()
    q = q.merge(df2)
    result = q.optimize(fuse=False)
    assert (
        len(list(node for node in result.walk() if isinstance(node, DiskShuffle))) == 2
    )

    expected = pdf.groupby("a").sum().reset_index()
    expected = expected.merge(pdf2)
    assert_eq(q, expected, check_index=False)

    q = df2.groupby("a").sum(split_out=True).reset_index()
    q = q.merge(df)
    result = q.optimize(fuse=False)
    # npartitions don't match in merge, so have to shuffle
    assert (
        len(list(node for node in result.walk() if isinstance(node, DiskShuffle))) == 3
    )
