import pickle
import sys

import pytest
from dask.dataframe._compat import PANDAS_GE_200

from dask_expr import new_collection
from dask_expr._expr import Lengths
from dask_expr.datasets import Timeseries, timeseries
from dask_expr.tests._util import assert_eq


def test_timeseries():
    df = timeseries(freq="360 s", start="2000-01-01", end="2000-01-02")
    assert_eq(df, df)


def test_optimization():
    df = timeseries(dtypes={"x": int, "y": float}, seed=123)
    expected = timeseries(dtypes={"x": int}, seed=123)
    result = df[["x"]].optimize(fuse=False)
    assert result.expr.frame.operand("columns") == expected.expr.frame.operand(
        "columns"
    )

    expected = timeseries(dtypes={"x": int}, seed=123)["x"].simplify()
    result = df["x"].optimize(fuse=False)
    assert expected.expr.frame.operand("columns") == result.expr.frame.operand(
        "columns"
    )


def test_arrow_string_option():
    df = timeseries(dtypes={"x": object, "y": float}, seed=123)
    result = df.optimize(fuse=False)
    assert result.x.dtype == "string"
    assert result.x.compute().dtype == "string"


def test_column_projection_deterministic():
    df = timeseries(freq="1h", start="2000-01-01", end="2000-01-02", seed=123)
    result_id = df[["id"]].optimize()
    result_id_x = df[["id", "x"]].optimize()
    assert_eq(result_id["id"], result_id_x["id"])


def test_timeseries_culling():
    df = timeseries(dtypes={"x": int, "y": float}, seed=123)
    pdf = df.compute()
    offset = len(df.partitions[0].compute())
    df = (df[["x"]] + 1).partitions[1]
    df2 = df.optimize()

    # All tasks should be fused for the single output partition
    assert df2.npartitions == 1
    assert len(df2.dask) == df2.npartitions
    expected = pdf.iloc[offset : 2 * offset][["x"]] + 1
    assert_eq(df2, expected)


def test_persist():
    df = timeseries(freq="1h", start="2000-01-01", end="2000-01-02", seed=123)
    a = df["x"]
    b = a.persist()

    assert_eq(a, b)
    assert len(b.dask) == 2 * b.npartitions


def test_lengths():
    df = timeseries(freq="1h", start="2000-01-01", end="2000-01-03", seed=123)
    assert len(df) == sum(new_collection(Lengths(df.expr).optimize()).compute())


def test_timeseries_empty_projection():
    ts = timeseries(end="2000-01-02", dtypes={})
    expected = timeseries(end="2000-01-02")
    assert len(ts) == len(expected)


def test_combine_similar(tmpdir):
    df = timeseries(end="2000-01-02")
    pdf = df.compute()
    got = df[df["name"] == "a"][["id"]]

    expected = pdf[pdf["name"] == "a"][["id"]]
    assert_eq(got, expected)
    assert_eq(got.optimize(fuse=False), expected)
    assert_eq(got.optimize(fuse=True), expected)

    # We should only have one Timeseries node, and
    # it should not include "z" in the dtypes
    timeseries_nodes = list(got.optimize(fuse=False).find_operations(Timeseries))
    assert len(timeseries_nodes) == 1
    assert set(timeseries_nodes[0].dtypes.keys()) == {"id", "name"}

    df = timeseries(end="2000-01-02")
    df2 = timeseries(end="2000-01-02")

    got = df + df2
    timeseries_nodes = list(got.optimize(fuse=False).find_operations(Timeseries))
    assert len(timeseries_nodes) == 2
    with pytest.raises(AssertionError):
        assert_eq(df + df2, 2 * df)


@pytest.mark.parametrize("seed", [42, None])
def test_timeseries_deterministic_head(seed):
    # Make sure our `random_state` code gives
    # us deterministic results
    df = timeseries(end="2000-01-02", seed=seed)
    assert_eq(df.head(), df.head())
    assert_eq(df["x"].head(), df.head()["x"])
    assert_eq(df.head()["x"], df["x"].partitions[0].compute().head())


@pytest.mark.parametrize("seed", [42, None])
def test_timeseries_gaph_size(seed):
    from dask.datasets import timeseries as dd_timeseries

    # Check that our graph size is reasonable
    df = timeseries(seed=seed)
    ddf = dd_timeseries(seed=seed)
    graph_size = sys.getsizeof(pickle.dumps(df.dask))
    graph_size_dd = sys.getsizeof(pickle.dumps(dict(ddf.dask)))
    # Make sure we are close to the dask.dataframe graph size
    threshold = 1.10 if PANDAS_GE_200 else 1.50
    assert graph_size < threshold * graph_size_dd


def test_dataset_head():
    ddf = timeseries(freq="1d")
    expected = ddf.compute()
    assert_eq(ddf.head(30, npartitions=-1), expected)
    assert_eq(ddf.head(30, npartitions=-1, compute=False), expected)
