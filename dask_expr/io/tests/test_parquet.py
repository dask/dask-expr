import os

import pandas as pd
import pyarrow as pa
import pytest
from dask.dataframe.utils import assert_eq

from dask_expr import from_pandas, optimize, read_parquet
from dask_expr._expr import Lengths, Literal
from dask_expr._reductions import Len
from dask_expr.io import ReadParquet


def _make_file(dir, df=None):
    fn = os.path.join(str(dir), f"myfile.parquet")
    if df is None:
        df = pd.DataFrame({c: range(10) for c in "abcde"})
    df.to_parquet(fn)
    return fn


@pytest.fixture
def parquet_file(tmpdir):
    return _make_file(tmpdir)


def test_parquet_len(tmpdir):
    df = read_parquet(_make_file(tmpdir))
    pdf = df.compute()

    assert len(df[df.a > 5]) == len(pdf[pdf.a > 5])

    s = (df["b"] + 1).astype("Int32")
    assert len(s) == len(pdf)

    assert isinstance(Len(s.expr).optimize(), Literal)
    assert isinstance(Lengths(s.expr).optimize(), Literal)


def test_parquet_len_filter(tmpdir):
    df = read_parquet(_make_file(tmpdir))
    expr = Len(df[df.c > 0].expr)
    result = expr.simplify()
    for rp in result.find_operations(ReadParquet):
        assert rp.operand("columns") == ["c"] or rp.operand("columns") == []


@pytest.mark.parametrize("write_metadata_file", [True, False])
def test_to_parquet(tmpdir, write_metadata_file):
    pdf = pd.DataFrame({"x": [1, 4, 3, 2, 0, 5]})
    df = from_pandas(pdf, npartitions=2)

    # Check basic parquet round trip
    df.to_parquet(tmpdir, write_metadata_file=write_metadata_file)
    df2 = read_parquet(tmpdir, calculate_divisions=True)
    assert_eq(df, df2)

    # Check overwrite behavior
    df["new"] = df["x"] + 1
    df.to_parquet(tmpdir, overwrite=True, write_metadata_file=write_metadata_file)
    df2 = read_parquet(tmpdir, calculate_divisions=True)
    assert_eq(df, df2)

    # Check that we cannot overwrite a path we are
    # reading from in the same graph
    with pytest.raises(ValueError, match="Cannot overwrite"):
        df2.to_parquet(tmpdir, overwrite=True)


def test_to_parquet_engine(tmpdir):
    pdf = pd.DataFrame({"x": [1, 4, 3, 2, 0, 5]})
    df = from_pandas(pdf, npartitions=2)
    with pytest.raises(NotImplementedError, match="not supported"):
        df.to_parquet(tmpdir + "engine.parquet", engine="fastparquet")


@pytest.mark.skip()
def test_predicate_pushdown(tmpdir):
    original = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5] * 10,
            "b": [0, 1, 2, 3, 4] * 10,
            "c": range(50),
            "d": [6, 7] * 25,
            "e": [8, 9] * 25,
        }
    )
    fn = _make_file(tmpdir, df=original)
    df = read_parquet(fn)
    assert_eq(df, original)
    x = df[df.a == 5][df.c > 20]["b"]
    y = optimize(x, fuse=False)
    assert isinstance(y.expr, ReadParquet)
    assert ("a", "==", 5) in y.expr.operand("filters")[0]
    assert ("c", ">", 20) in y.expr.operand("filters")[0]
    assert list(y.columns) == ["b"]

    # Check computed result
    y_result = y.compute()
    assert y_result.name == "b"
    assert len(y_result) == 6
    assert (y_result == 4).all()


@pytest.mark.skip()
def test_predicate_pushdown_compound(tmpdir):
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5] * 10,
            "b": [0, 1, 2, 3, 4] * 10,
            "c": range(50),
            "d": [6, 7] * 25,
            "e": [8, 9] * 25,
        }
    )
    fn = _make_file(tmpdir, df=pdf)
    df = read_parquet(fn)

    # Test AND
    x = df[(df.a == 5) & (df.c > 20)]["b"]
    y = optimize(x, fuse=False)
    assert isinstance(y.expr, ReadParquet)
    assert {("c", ">", 20), ("a", "==", 5)} == set(y.filters[0])
    assert_eq(
        y,
        pdf[(pdf.a == 5) & (pdf.c > 20)]["b"],
        check_index=False,
    )

    # Test OR
    x = df[(df.a == 5) | (df.c > 20)]
    x = x[x.b != 0]["b"]
    y = optimize(x, fuse=False)
    assert isinstance(y.expr, ReadParquet)
    filters = [set(y.filters[0]), set(y.filters[1])]
    assert {("c", ">", 20), ("b", "!=", 0)} in filters
    assert {("a", "==", 5), ("b", "!=", 0)} in filters
    expect = pdf[(pdf.a == 5) | (pdf.c > 20)]
    expect = expect[expect.b != 0]["b"]
    assert_eq(
        y,
        expect,
        check_index=False,
    )

    # Test OR and AND
    x = df[((df.a == 5) | (df.c > 20)) & (df.b != 0)]["b"]
    z = optimize(x, fuse=False)
    assert isinstance(z.expr, ReadParquet)
    filters = [set(z.filters[0]), set(z.filters[1])]
    assert {("c", ">", 20), ("b", "!=", 0)} in filters
    assert {("a", "==", 5), ("b", "!=", 0)} in filters
    assert_eq(y, z)


def test_pyarrow_filesystem(parquet_file):
    from pyarrow import fs

    fs = fs.LocalFileSystem()

    df_pa = read_parquet(parquet_file, filesystem=fs)
    df = read_parquet(parquet_file)
    assert assert_eq(df, df_pa)
