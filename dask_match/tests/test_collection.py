import pandas as pd
import pytest
from dask.dataframe.utils import assert_eq
from dask.utils import M

from dask_match import from_pandas, optimize


@pytest.fixture
def df():
    df = pd.DataFrame({"x": range(100)})
    df["y"] = df.x * 10.0
    yield df


@pytest.fixture
def ddf(df):
    yield from_pandas(df, npartitions=10)


def test_del():
    df = pd.DataFrame({"x": range(100), "y": range(100)})
    ddf = from_pandas(df, npartitions=10)
    df = df.copy()

    # Check __delitem__
    del df["x"]
    del ddf["x"]
    assert_eq(df, ddf)


def test_setitem():
    df = pd.DataFrame({"x": range(100), "y": range(100)})
    ddf = from_pandas(df, npartitions=10)
    df = df.copy()

    ddf["z"] = ddf.x + ddf.y

    assert "z" in ddf.columns
    assert_eq(ddf, ddf)


def test_meta_divisions_name():
    a = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    df = 2 * from_pandas(a, npartitions=2)
    assert list(df.columns) == list(a.columns)
    assert df.npartitions == 2

    assert df.x.sum()._meta == 0
    assert df.x.sum().npartitions == 1

    assert "mul" in df._name
    assert "sum" in df.sum()._name


def test_meta_blockwise():
    a = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    b = pd.DataFrame({"z": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})

    aa = from_pandas(a, npartitions=2)
    bb = from_pandas(b, npartitions=2)

    cc = 2 * aa - 3 * bb
    assert set(cc.columns) == {"x", "y", "z"}


def test_dask(df, ddf):
    assert (ddf.x + ddf.y).npartitions == 10
    z = (ddf.x + ddf.y).sum()

    assert assert_eq(z, (df.x + df.y).sum())


@pytest.mark.parametrize(
    "func",
    [
        M.max,
        M.min,
        M.sum,
        M.count,
        M.mean,
        pytest.param(
            lambda df: df.size,
            marks=pytest.mark.skip(reason="scalars don't work yet"),
        ),
    ],
)
def test_reductions(func, df, ddf):
    assert_eq(func(ddf), func(df))
    assert_eq(func(ddf.x), func(df.x))


def test_mode():
    df = pd.DataFrame({"x": [1, 2, 3, 1, 2]})
    ddf = from_pandas(df, npartitions=3)

    assert_eq(ddf.x.mode(), df.x.mode(), check_names=False)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.x > 10,
        lambda df: df.x + 20 > df.y,
        lambda df: 10 < df.x,
        lambda df: 10 <= df.x,
        lambda df: 10 == df.x,
        lambda df: df.x < df.y,
        lambda df: df.x > df.y,
        lambda df: df.x == df.y,
        lambda df: df.x != df.y,
    ],
)
def test_conditionals(func, df, ddf):
    assert_eq(func(df), func(ddf), check_names=False)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.astype(int),
        lambda df: df.apply(lambda row, x, y=10: row * x + y, x=2),
        lambda df: df[df.x > 5],
        lambda df: df.assign(a=df.x + df.y, b=df.x - df.y),
    ],
)
def test_blockwise(func, df, ddf):
    assert_eq(func(df), func(ddf))


def test_repr(ddf):
    assert "+ 1" in str(ddf + 1)
    assert "+ 1" in repr(ddf + 1)

    s = (ddf["x"] + 1).sum(skipna=False).expr
    assert '["x"]' in s or "['x']" in s
    assert "+ 1" in s
    assert "sum(skipna=False)" in s


def test_columns_traverse_filters(df, ddf):
    result = optimize(ddf[ddf.x > 5].y, fuse=False)
    expected = ddf.y[ddf.x > 5]

    assert str(result) == str(expected)


def test_optimize_fusion(ddf):
    ddf2 = (ddf["x"] + ddf["y"]) - 1
    unfused = optimize(ddf2, fuse=False)
    fused = optimize(ddf2, fuse=True)

    # Should only get one task per partition
    assert len(fused.dask) == ddf.npartitions
    assert_eq(fused, unfused)

    # Check that we still get fusion when
    # non-blockwise operations are added
    ddf3 = ddf2.sum()
    unfused = optimize(ddf3, fuse=False)
    fused = optimize(ddf3, fuse=True)

    assert len(fused.dask) < len(unfused.dask)
    assert_eq(fused, unfused)

    # Check that we still get fusion
    # after a non-blockwise operation as well
    fused_2 = optimize((ddf3 + 10) - 5, fuse=True)
    # The "+10 and -5" ops should get fused
    assert len(fused_2.dask) == len(fused.dask) + 1


@pytest.mark.parametrize("persist", [True, False])
def test_optimize_fusion_many(ddf, persist):
    # Test that many `Blockwise`` operations,
    # originating from various IO operations,
    # can all be fused together
    ddfa = ddf.persist() if persist else ddf
    ddfb = from_pandas(pd.DataFrame({"a": range(100)}), ddf.npartitions)

    ddf2a = ddfa[["x"]] + 1
    ddf2a["a"] = ddfa["y"] + ddfa["x"]
    ddf2a["b"] = ddf2a["x"] + 2
    sera = ddf2a[ddfa["x"] > 1]["b"]

    ddf2b = ddfb[["a"]] + 1
    ddf2b["b"] = ddfb["a"] + ddfb["a"]
    serb = ddf2b[ddfb["a"] > 1]["b"]

    ser = (sera + serb) + 1
    unfused = optimize(ser, fuse=False)
    fused = optimize(ser, fuse=True)
    if persist:
        # Can't fuse persisted graph, but everything
        # after should be fused
        assert len(fused.dask) == ddf.npartitions * 2
    else:
        # Everything should be fused
        assert len(fused.dask) == ddf.npartitions
    assert_eq(fused, unfused)


def test_optimize_fusion_repeat(ddf):
    # Test that we can optimize a collection
    # more than once, and fusion still works
    ddf2 = (ddf[["x"]] + 1).assign(z=ddf.y)
    ddf3 = ddf2 + 2
    ddf3_fused = optimize(ddf2, fuse=True) + 2

    fused = optimize(ddf3_fused, fuse=True)
    assert len(fused.dask) == ddf.npartitions
    assert_eq(fused, ddf3)

    ser = ddf3["x"]
    ser_fused = optimize(fused["x"], fuse=True)
    ser_fused = optimize(ser_fused, fuse=True)
    assert len(ser_fused.dask) == ddf.npartitions
    assert_eq(ser_fused, ser)


def test_persist(df, ddf):
    a = ddf + 2
    b = a.persist()

    assert_eq(a, b)
    assert len(a.__dask_graph__()) > len(b.__dask_graph__())

    assert len(b.__dask_graph__()) == b.npartitions

    assert_eq(b.y.sum(), (df + 2).y.sum())


def test_persist_with_fusion(ddf):
    # Check that fusion works after persisting
    a = ddf + 2
    b = a.persist()

    c = (b.y + 1).sum()
    c_fused = optimize(c, fuse=True)
    assert_eq(c, c_fused)
    assert len(c_fused.dask) < len(c.dask)


def test_index(df, ddf):
    assert_eq(ddf.index, df.index)
    assert_eq(ddf.x.index, df.x.index)


def test_head(df, ddf):
    assert_eq(ddf.head(compute=False), df.head())
    assert_eq(ddf.head(compute=False, n=7), df.head(n=7))

    assert ddf.head(compute=False).npartitions == 1


def test_substitute(ddf):
    pdf = pd.DataFrame(
        {
            "a": range(100),
            "b": range(100),
            "c": range(100),
        }
    )
    df = from_pandas(pdf, npartitions=3)
    df = df.expr

    result = (df + 1).substitute({1: 2})
    expected = df + 2
    assert result._name == expected._name

    result = df["a"].substitute({df["a"]: df["b"]})
    expected = df["b"]
    assert result._name == expected._name

    result = (df["a"] - df["b"]).substitute({df["b"]: df["c"]})
    expected = df["a"] - df["c"]
    assert result._name == expected._name

    result = df["a"].substitute({3: 4})
    expected = from_pandas(pdf, npartitions=4).a
    assert result._name == expected._name

    result = (df["a"].sum() + 5).substitute({df["a"]: df["b"], 5: 6})
    expected = df["b"].sum() + 6
    assert result._name == expected._name
