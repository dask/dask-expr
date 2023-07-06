from __future__ import annotations

import operator
import pickle
import re

import dask
import numpy as np
import pandas as pd
import pytest
from dask.dataframe._compat import PANDAS_GT_210
from dask.dataframe.utils import UNKNOWN_CATEGORIES, assert_eq
from dask.utils import M

from dask_expr import expr, from_pandas, optimize
from dask_expr.datasets import timeseries
from dask_expr.expr import are_co_aligned
from dask_expr.reductions import Len

try:
    import cudf
except ImportError:
    cudf = None


@pytest.fixture(
    params=[
        "pandas",
        pytest.param(
            "cudf", marks=pytest.mark.skipif(cudf is None, reason="cudf not found.")
        ),
    ]
)
def backend(request):
    yield request.param


@pytest.fixture
def lib(backend):
    # Multi-backend DataFrame fixture
    if backend == "cudf":
        yield cudf
    else:
        yield pd


@pytest.fixture
def bdf(lib):
    # Backend DataFrame fixture
    df = lib.DataFrame({"x": range(100)})
    df["y"] = df.x * 10.0
    yield df


@pytest.fixture
def xdf(bdf):
    # Multi-backend Dask-Expression DataFrame fixture
    yield from_pandas(bdf, npartitions=10)


def test_del(bdf, xdf):
    pdf = bdf.copy()

    # Check __delitem__
    del pdf["x"]
    del xdf["x"]
    assert_eq(pdf, xdf)


def test_setitem(bdf, xdf):
    pdf = bdf.copy()
    pdf["z"] = pdf.x + pdf.y

    xdf["z"] = xdf.x + xdf.y

    assert "z" in xdf.columns
    assert_eq(xdf, pdf)


def test_explode():
    # CuDF backend does not support explode
    # (See: https://github.com/rapidsai/cudf/issues/10271)
    pdf = pd.DataFrame({"a": [[1, 2], [3, 4]]})
    df = from_pandas(pdf)
    assert_eq(pdf.explode(column="a"), df.explode(column="a"))
    assert_eq(pdf.a.explode(), df.a.explode())


def test_explode_simplify(bdf, backend):
    if backend == "cudf":
        pytest.xfail(reason="https://github.com/rapidsai/cudf/issues/10271")
    pdf = bdf.copy()
    pdf["z"] = 1
    df = from_pandas(pdf)
    q = df.explode(column="x")["y"]
    result = optimize(q, fuse=False)
    expected = df[["x", "y"]].explode(column="x")["y"]
    assert result._name == expected._name


def test_meta_divisions_name(lib):
    a = lib.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    df = 2 * from_pandas(a, npartitions=2)
    assert list(df.columns) == list(a.columns)
    assert df.npartitions == 2

    assert np.isscalar(df.x.sum()._meta)
    assert df.x.sum().npartitions == 1

    assert "mul" in df._name
    assert "sum" in df.sum()._name


def test_meta_blockwise(lib):
    a = lib.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    b = lib.DataFrame({"z": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})

    aa = from_pandas(a, npartitions=2)
    bb = from_pandas(b, npartitions=2)

    cc = 2 * aa - 3 * bb
    assert set(cc.columns) == {"x", "y", "z"}


def test_dask(bdf, xdf):
    assert (xdf.x + xdf.y).npartitions == 10
    z = (xdf.x + xdf.y).sum()

    assert assert_eq(z, (bdf.x + bdf.y).sum())


@pytest.mark.parametrize(
    "func",
    [
        M.max,
        M.min,
        M.any,
        M.all,
        M.sum,
        M.prod,
        M.count,
        M.mean,
        M.idxmin,
        M.idxmax,
        pytest.param(
            lambda df: df.size,
            marks=pytest.mark.skip(reason="scalars don't work yet"),
        ),
    ],
)
def test_reductions(func, bdf, xdf, backend):
    if backend == "cudf" and func in [M.idxmin, M.idxmax]:
        pytest.xfail(reason="https://github.com/rapidsai/cudf/issues/9602")
    result = func(xdf)
    assert result.known_divisions
    assert_eq(result, func(bdf))
    result = func(xdf.x)
    assert not result.known_divisions
    assert_eq(result, func(bdf.x))
    # check_dtype False because sub-selection of columns that is pushed through
    # is not reflected in the meta calculation
    assert_eq(func(xdf)["x"], func(bdf)["x"], check_dtype=False)


def test_nbytes(bdf, xdf, backend):
    if backend == "cudf":
        pytest.xfail(reason="nbytes not supported by cudf")
    with pytest.raises(NotImplementedError, match="nbytes is not implemented"):
        xdf.nbytes
    assert_eq(xdf.x.nbytes, bdf.x.nbytes)


def test_mode(lib):
    pdf = lib.DataFrame({"x": [1, 2, 3, 1, 2]})
    df = from_pandas(pdf, npartitions=3)

    assert_eq(df.x.mode(), pdf.x.mode(), check_names=False)


def test_value_counts(xdf, bdf):
    with pytest.raises(
        AttributeError, match="'DataFrame' object has no attribute 'value_counts'"
    ):
        xdf.value_counts()
    assert_eq(xdf.x.value_counts(), bdf.x.value_counts().astype("int64"))


def test_dropna(bdf):
    pdf = bdf.copy()
    pdf.loc[0, "y"] = np.nan
    df = from_pandas(pdf)
    assert_eq(df.dropna(), pdf.dropna())
    assert_eq(df.dropna(how="all"), pdf.dropna(how="all"))
    assert_eq(df.y.dropna(), pdf.y.dropna())


def test_memory_usage(bdf):
    # Results are not equal with RangeIndex because pandas has one RangeIndex while
    # we have one RangeIndex per partition
    pdf = bdf.copy()
    pdf.index = np.arange(len(pdf))
    df = from_pandas(pdf)
    assert_eq(df.memory_usage(), pdf.memory_usage())
    assert_eq(df.memory_usage(index=False), pdf.memory_usage(index=False))
    assert_eq(df.x.memory_usage(), pdf.x.memory_usage())
    assert_eq(df.x.memory_usage(index=False), pdf.x.memory_usage(index=False))
    assert_eq(df.index.memory_usage(), pdf.index.memory_usage())
    with pytest.raises(TypeError, match="got an unexpected keyword"):
        df.index.memory_usage(index=True)


@pytest.mark.parametrize("func", [M.nlargest, M.nsmallest])
def test_nlargest_nsmallest(xdf, bdf, func):
    assert_eq(func(xdf, n=5, columns="x"), func(bdf, n=5, columns="x"))
    assert_eq(func(xdf.x, n=5), func(bdf.x, n=5))
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        func(xdf.x, n=5, columns="foo")


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
def test_conditionals(func, bdf, xdf):
    assert_eq(func(bdf), func(xdf), check_names=False)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.x & df.y,
        lambda df: df.x.__rand__(df.y),
        lambda df: df.x | df.y,
        lambda df: df.x.__ror__(df.y),
        lambda df: df.x ^ df.y,
        lambda df: df.x.__rxor__(df.y),
    ],
)
def test_boolean_operators(func, lib):
    pdf = lib.DataFrame(
        {"x": [True, False, True, False], "y": [True, False, False, False]}
    )
    df = from_pandas(pdf)
    assert_eq(func(pdf), func(df))


@pytest.mark.parametrize(
    "func",
    [
        lambda df: ~df,
        lambda df: ~df.x,
        lambda df: -df.z,
        lambda df: +df.z,
        lambda df: -df,
        lambda df: +df,
    ],
)
def test_unary_operators(func, lib):
    pdf = lib.DataFrame(
        {"x": [True, False, True, False], "y": [True, False, False, False], "z": 1}
    )
    df = from_pandas(pdf)
    assert_eq(func(pdf), func(df))


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df[(df.x > 10) | (df.x < 5)],
        lambda df: df[(df.x > 7) & (df.x < 10)],
    ],
)
def test_and_or(func, bdf, xdf):
    assert_eq(func(bdf), func(xdf), check_names=False)


@pytest.mark.parametrize("how", ["start", "end"])
def test_to_timestamp(bdf, how, backend):
    if backend == "cudf":
        pytest.xfail(reason="period_range not supported by cudf")
    bdf.index = pd.period_range("2019-12-31", freq="D", periods=len(bdf))
    df = from_pandas(bdf)
    assert_eq(df.to_timestamp(how=how), bdf.to_timestamp(how=how))
    assert_eq(df.x.to_timestamp(how=how), bdf.x.to_timestamp(how=how))


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.astype(int),
        # lambda df: df.apply(lambda row, x, y=10: row * x + y, x=2),
        pytest.param(
            lambda df: df.map(lambda x: x + 1),
            marks=pytest.mark.skipif(
                not PANDAS_GT_210, reason="Only available from 2.1"
            ),
        ),
        lambda df: df.clip(lower=10, upper=50),
        lambda df: df.x.clip(lower=10, upper=50),
        lambda df: df.x.between(left=10, right=50),
        lambda df: df.x.map(lambda x: x + 1),
        # lambda df: df.index.map(lambda x: x + 1),
        lambda df: df[df.x > 5],
        lambda df: df.assign(a=df.x + df.y, b=df.x - df.y),
        lambda df: df.replace(to_replace=1, value=1000),
        lambda df: df.x.replace(to_replace=1, value=1000),
        lambda df: df.isna(),
        lambda df: df.x.isna(),
        lambda df: df.abs(),
        lambda df: df.x.abs(),
        lambda df: df.rename(columns={"x": "xx"}),
        lambda df: df.rename(columns={"x": "xx"}).xx,
        lambda df: df.rename(columns={"x": "xx"})[["xx"]],
        # lambda df: df.combine_first(df),
        # lambda df: df.x.combine_first(df.y),
        lambda df: df.x.to_frame(),
        lambda df: df.drop(columns="x"),
        lambda df: df.x.index.to_frame(),
        lambda df: df.eval("z=x+y"),
        lambda df: df.select_dtypes(include="integer"),
    ],
)
def test_blockwise(func, bdf, xdf):
    assert_eq(func(bdf), func(xdf))


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.apply(lambda row, x, y=10: row * x + y, x=2),
        lambda df: df.index.map(lambda x: x + 1),
        lambda df: df.combine_first(df),
        lambda df: df.x.combine_first(df.y),
    ],
)
def test_blockwise_cudf_fails(func, bdf, xdf, backend):
    if backend == "cudf":
        pytest.xfail(reason="func not supported by cudf")
    assert_eq(func(bdf), func(xdf))


def test_rename_axis(bdf, backend):
    if backend == "cudf":
        pytest.xfail(reason="rename_axis not supported by cudf")
    pdf = bdf.copy()
    pdf.index.name = "a"
    pdf.columns.name = "b"
    df = from_pandas(pdf, npartitions=10)
    assert_eq(df.rename_axis(index="dummy"), pdf.rename_axis(index="dummy"))
    assert_eq(df.rename_axis(columns="dummy"), pdf.rename_axis(columns="dummy"))
    assert_eq(df.x.rename_axis(index="dummy"), pdf.x.rename_axis(index="dummy"))


def test_isin(xdf, bdf):
    values = [1, 2]
    assert_eq(bdf.isin(values), xdf.isin(values))
    assert_eq(bdf.x.isin(values), xdf.x.isin(values))


def test_round(bdf):
    pdf = bdf.copy()
    pdf += 0.5555
    df = from_pandas(pdf)
    assert_eq(df.round(decimals=1), pdf.round(decimals=1))
    assert_eq(df.x.round(decimals=1), pdf.x.round(decimals=1))


def test_repr(xdf):
    assert "+ 1" in str(xdf + 1)
    assert "+ 1" in repr(xdf + 1)

    s = (xdf["x"] + 1).sum(skipna=False).expr
    assert '["x"]' in s or "['x']" in s
    assert "+ 1" in s
    assert "sum(skipna=False)" in s


def test_combine_first_simplify(bdf, backend):
    if backend == "cudf":
        pytest.xfail(reason="combine_first not supported by cudf")
    pdf = bdf.copy()
    df = from_pandas(pdf)
    pdf2 = pdf.rename(columns={"y": "z"})
    df2 = from_pandas(pdf2)

    q = df.combine_first(df2)[["z", "y"]]
    result = q.simplify()
    expected = df[["y"]].combine_first(df2[["z"]])[["z", "y"]]
    assert result._name == expected._name
    assert_eq(result, pdf.combine_first(pdf2)[["z", "y"]])


def test_rename_traverse_filter(xdf):
    result = xdf.rename(columns={"x": "xx"})[["xx"]].simplify()
    expected = xdf[["x"]].rename(columns={"x": "xx"})
    assert str(result) == str(expected)


def test_columns_traverse_filters(xdf):
    result = xdf[xdf.x > 5].y.simplify()
    expected = xdf.y[xdf.x > 5]

    assert str(result) == str(expected)


def test_clip_traverse_filters(xdf):
    result = xdf.clip(lower=10).y.simplify()
    expected = xdf.y.clip(lower=10)

    assert result._name == expected._name

    result = xdf.clip(lower=10)[["x", "y"]].simplify()
    expected = xdf.clip(lower=10)

    assert result._name == expected._name

    arg = xdf.clip(lower=10)[["x"]]
    result = arg.simplify()
    expected = xdf[["x"]].clip(lower=10)

    assert result._name == expected._name


@pytest.mark.parametrize("projection", ["zz", ["zz"], ["zz", "x"], "zz"])
@pytest.mark.parametrize("subset", ["x", ["x"]])
def test_drop_duplicates_subset_simplify(bdf, subset, projection):
    pdf = bdf.copy()
    pdf["zz"] = 1
    df = from_pandas(pdf)
    result = df.drop_duplicates(subset=subset)[projection].simplify()
    expected = df[["x", "zz"]].drop_duplicates(subset=subset)[projection]

    assert str(result) == str(expected)


def test_broadcast(bdf, xdf):
    assert_eq(
        xdf + xdf.sum(),
        bdf + bdf.sum(),
    )
    assert_eq(
        xdf.x + xdf.x.sum(),
        bdf.x + bdf.x.sum(),
    )


def test_persist(bdf, xdf):
    a = xdf + 2
    b = a.persist()

    assert_eq(a, b)
    assert len(a.__dask_graph__()) > len(b.__dask_graph__())

    assert len(b.__dask_graph__()) == b.npartitions

    assert_eq(b.y.sum(), (bdf + 2).y.sum())


def test_index(bdf, xdf):
    assert_eq(xdf.index, bdf.index)
    assert_eq(xdf.x.index, bdf.x.index)


@pytest.mark.parametrize("drop", [True, False])
def test_reset_index(bdf, xdf, drop):
    assert_eq(xdf.reset_index(drop=drop), bdf.reset_index(drop=drop), check_index=False)
    assert_eq(
        xdf.x.reset_index(drop=drop), bdf.x.reset_index(drop=drop), check_index=False
    )


def test_head(bdf, xdf):
    assert_eq(xdf.head(compute=False), bdf.head())
    assert_eq(xdf.head(compute=False, n=7), bdf.head(n=7))

    assert xdf.head(compute=False).npartitions == 1


def test_head_down(xdf):
    result = (xdf.x + xdf.y + 1).head(compute=False)
    optimized = result.simplify()

    assert_eq(result, optimized)

    assert not isinstance(optimized.expr, expr.Head)


def test_head_head(xdf):
    a = xdf.head(compute=False).head(compute=False)
    b = xdf.head(compute=False)

    assert a.optimize()._name == b.optimize()._name


def test_tail(bdf, xdf):
    assert_eq(xdf.tail(compute=False), bdf.tail())
    assert_eq(xdf.tail(compute=False, n=7), bdf.tail(n=7))

    assert xdf.tail(compute=False).npartitions == 1


def test_tail_down(xdf):
    result = (xdf.x + xdf.y + 1).tail(compute=False)
    optimized = optimize(result)

    assert_eq(result, optimized)

    assert not isinstance(optimized.expr, expr.Tail)


def test_tail_tail(xdf):
    a = xdf.tail(compute=False).tail(compute=False)
    b = xdf.tail(compute=False)

    assert a.optimize()._name == b.optimize()._name


def test_tail_repartition(xdf):
    a = xdf.repartition(npartitions=10).tail()
    b = xdf.tail()
    assert_eq(a, b)


def test_projection_stacking(xdf):
    result = xdf[["x", "y"]]["x"]
    optimized = result.simplify()
    expected = xdf["x"]

    assert optimized._name == expected._name


def test_projection_stacking_coercion(bdf):
    df = from_pandas(bdf)
    assert_eq(df.x[0], bdf.x[0], check_divisions=False)
    assert_eq(df.x[[0]], bdf.x[[0]], check_divisions=False)


def test_remove_unnecessary_projections(xdf):
    result = (xdf + 1)[xdf.columns]
    optimized = result.simplify()
    expected = xdf + 1

    assert optimized._name == expected._name

    result = (xdf[["x"]] + 1)[["x"]]
    optimized = result.simplify()
    expected = xdf[["x"]] + 1

    assert optimized._name == expected._name


def test_substitute(lib):
    pdf = lib.DataFrame(
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


def test_from_pandas(bdf):
    df = from_pandas(bdf, npartitions=3)
    assert df.npartitions == 3
    assert "pandas" in df._name


def test_copy(xdf):
    original = xdf.copy()
    columns = tuple(original.columns)

    xdf["z"] = xdf.x + xdf.y

    assert tuple(original.columns) == columns
    assert "z" not in original.columns


def test_partitions(bdf, xdf):
    assert_eq(xdf.partitions[0], bdf.iloc[:10])
    assert_eq(xdf.partitions[1], bdf.iloc[10:20])
    assert_eq(xdf.partitions[1:3], bdf.iloc[10:30])
    assert_eq(xdf.partitions[[3, 4]], bdf.iloc[30:50])
    assert_eq(xdf.partitions[-1], bdf.iloc[90:])

    out = (xdf + 1).partitions[0].simplify()
    assert isinstance(out.expr, expr.Add)
    assert out.expr.left._partitions == [0]

    # Check culling
    out = optimize(xdf.partitions[1])
    assert len(out.dask) == 1
    assert_eq(out, bdf.iloc[10:20])


def test_column_getattr(xdf):
    xdf = xdf.expr
    assert xdf.x._name == xdf["x"]._name

    with pytest.raises(AttributeError):
        xdf.foo


def test_serialization(bdf, xdf):
    before = pickle.dumps(xdf)

    assert len(before) < 200 + len(pickle.dumps(bdf))

    part = xdf.partitions[0].compute()
    assert (
        len(pickle.dumps(xdf.__dask_graph__()))
        < 1000 + len(pickle.dumps(part)) * xdf.npartitions
    )

    after = pickle.dumps(xdf)

    assert before == after  # caching doesn't affect serialization

    assert pickle.loads(before)._name == pickle.loads(after)._name
    assert_eq(pickle.loads(before), pickle.loads(after))


def test_size_optimized(xdf, backend):
    if backend == "cudf":
        pytest.xfail(reason="Cannot apply lambda function in cudf")
    expr = (xdf.x + 1).apply(lambda x: x).size
    out = optimize(expr)
    expected = optimize(xdf.x.size)
    assert out._name == expected._name

    expr = (xdf + 1).apply(lambda x: x).size
    out = optimize(expr)
    expected = optimize(xdf.size)
    assert out._name == expected._name


@pytest.mark.parametrize("fuse", [True, False])
def test_tree_repr(fuse):
    s = from_pandas(pd.Series(range(10))).expr.tree_repr()
    assert "<pandas>" in s

    df = timeseries()
    expr = ((df.x + 1).sum(skipna=False) + df.y.mean()).expr
    expr = expr.optimize() if fuse else expr
    s = expr.tree_repr()

    assert "Sum" in s
    assert "Add" in s
    assert "1" in s
    assert "True" not in s
    assert "None" not in s
    assert "skipna=False" in s
    assert str(df.seed) in s.lower()
    if fuse:
        assert "Fused" in s
        assert s.count("|") == 9


def test_simple_graphs(xdf):
    expr = (xdf + 1).expr
    graph = expr.__dask_graph__()

    assert graph[(expr._name, 0)] == (operator.add, (xdf.expr._name, 0), 1)


def test_map_partitions(xdf):
    def combine_x_y(x, y, foo=None):
        assert foo == "bar"
        return x + y

    df2 = xdf.map_partitions(combine_x_y, xdf + 1, foo="bar")
    assert_eq(df2, xdf + (xdf + 1))


def test_map_partitions_broadcast(xdf):
    def combine_x_y(x, y, val, foo=None):
        assert foo == "bar"
        return x + y + val

    df2 = xdf.map_partitions(combine_x_y, xdf["x"].sum(), 123, foo="bar")
    assert_eq(df2, xdf + xdf["x"].sum() + 123)
    assert_eq(df2.optimize(), xdf + xdf["x"].sum() + 123)


@pytest.mark.parametrize("opt", [True, False])
def test_map_partitions_merge(opt, lib):
    # Make simple left & right dfs
    pdf1 = lib.DataFrame({"x": range(20), "y": range(20)})
    df1 = from_pandas(pdf1, 2)
    pdf2 = lib.DataFrame({"x": range(0, 20, 2), "z": range(10)})
    df2 = from_pandas(pdf2, 1)

    # Partition-wise merge with map_partitions
    df3 = df1.map_partitions(
        lambda l, r: l.merge(r, on="x"),
        df2,
        enforce_metadata=False,
        clear_divisions=True,
    )

    # Check result with/without fusion
    expect = pdf1.merge(pdf2, on="x")
    df3 = (df3.optimize() if opt else df3)[list(expect.columns)]
    assert_eq(df3, expect, check_index=False)


def test_depth(xdf):
    assert xdf._depth() == 1
    assert (xdf + 1)._depth() == 2
    assert ((xdf.x + 1) + xdf.y)._depth() == 4


def test_partitions_nested(xdf):
    a = expr.Partitions(expr.Partitions(xdf.expr, [2, 4, 6]), [0, 2])
    b = expr.Partitions(xdf.expr, [2, 6])

    assert a.optimize()._name == b.optimize()._name


@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("npartitions", [7, 12])
def test_repartition_npartitions(bdf, npartitions, sort):
    df = from_pandas(bdf, sort=sort) + 1
    df2 = df.repartition(npartitions=npartitions)
    assert df2.npartitions == npartitions
    assert_eq(df, df2)


@pytest.mark.parametrize("opt", [True, False])
def test_repartition_divisions(xdf, opt):
    end = xdf.divisions[-1] + 100
    stride = end // (xdf.npartitions + 2)
    divisions = tuple(range(0, end, stride))
    df2 = (xdf + 1).repartition(divisions=divisions, force=True)["x"]
    df2 = optimize(df2) if opt else df2
    assert df2.divisions == divisions
    assert_eq((xdf + 1)["x"], df2)

    # Check partitions
    for p, part in enumerate(dask.compute(list(df2.index.partitions))[0]):
        if len(part):
            assert part.min() >= df2.divisions[p]
            assert part.max() < df2.divisions[p + 1]


def test_repartition_no_op(xdf):
    result = xdf.repartition(divisions=xdf.divisions).optimize()
    assert result._name == xdf._name


def test_len(xdf, bdf):
    df2 = xdf[["x"]] + 1
    assert len(df2) == len(bdf)

    assert len(xdf[xdf.x > 5]) == len(bdf[bdf.x > 5])

    first = df2.partitions[0].compute()
    assert len(df2.partitions[0]) == len(first)

    assert isinstance(Len(df2.expr).optimize(), expr.Literal)
    assert isinstance(expr.Lengths(df2.expr).optimize(), expr.Literal)


def test_astype_simplify(xdf, bdf):
    q = xdf.astype({"x": "float64", "y": "float64"})["x"]
    result = q.simplify()
    expected = xdf["x"].astype({"x": "float64"})
    assert result._name == expected._name
    assert_eq(q, bdf.astype({"x": "float64", "y": "float64"})["x"])

    q = xdf.astype({"y": "float64"})["x"]
    result = q.simplify()
    expected = xdf["x"]
    assert result._name == expected._name

    q = xdf.astype("float64")["x"]
    result = q.simplify()
    expected = xdf["x"].astype("float64")
    assert result._name == expected._name


def test_drop_duplicates(xdf, bdf, backend):
    assert_eq(xdf.drop_duplicates(), bdf.drop_duplicates())
    assert_eq(
        xdf.drop_duplicates(ignore_index=True), bdf.drop_duplicates(ignore_index=True)
    )
    assert_eq(xdf.drop_duplicates(subset=["x"]), bdf.drop_duplicates(subset=["x"]))
    assert_eq(xdf.x.drop_duplicates(), bdf.x.drop_duplicates())

    if backend == "pandas":
        with pytest.raises(KeyError, match=re.escape("Index(['a'], dtype='object')")):
            xdf.drop_duplicates(subset=["a"])

    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        xdf.x.drop_duplicates(subset=["a"])


def test_unique(xdf, bdf, lib):
    with pytest.raises(
        AttributeError, match="'DataFrame' object has no attribute 'unique'"
    ):
        xdf.unique()

    # pandas returns a numpy array while we return a Series/Index
    assert_eq(xdf.x.unique(), lib.Series(bdf.x.unique(), name="x"))
    assert_eq(xdf.index.unique(), lib.Index(bdf.index.unique()))


def test_walk(xdf):
    df2 = xdf[xdf["x"] > 1][["y"]] + 1
    assert all(isinstance(ex, expr.Expr) for ex in df2.walk())
    exprs = set(df2.walk())
    assert xdf.expr in exprs
    assert xdf["x"].expr in exprs
    assert (xdf["x"] > 1).expr in exprs
    assert 1 not in exprs


def test_find_operations(xdf):
    df2 = xdf[xdf["x"] > 1][["y"]] + 1

    filters = list(df2.find_operations(expr.Filter))
    assert len(filters) == 1

    projections = list(df2.find_operations(expr.Projection))
    assert len(projections) == 2

    adds = list(df2.find_operations(expr.Add))
    assert len(adds) == 1
    assert next(iter(adds))._name == df2._name

    both = list(df2.find_operations((expr.Add, expr.Filter)))
    assert len(both) == 2


@pytest.mark.parametrize("subset", ["x", ["x"]])
def test_dropna_simplify(bdf, subset):
    bdf["z"] = 1
    df = from_pandas(bdf)
    q = df.dropna(subset=subset)["y"]
    result = q.simplify()
    expected = df[["x", "y"]].dropna(subset=subset)["y"]
    assert result._name == expected._name
    assert_eq(q, bdf.dropna(subset=subset)["y"])


def test_dir(xdf):
    assert all(c in dir(xdf) for c in xdf.columns)
    assert "sum" in dir(xdf)
    assert "sum" in dir(xdf.x)
    assert "sum" in dir(xdf.index)


@pytest.mark.parametrize(
    "func, args",
    [
        ("replace", (1, 2)),
        ("isin", ([1, 2],)),
        ("clip", (0, 5)),
        ("isna", ()),
        ("round", ()),
        ("abs", ()),
        # ("map", (lambda x: x+1, )),  # add in when pandas 2.1 is out
    ],
)
@pytest.mark.parametrize("indexer", ["x", ["x"]])
def test_simplify_up_blockwise(xdf, bdf, func, args, indexer):
    q = getattr(xdf, func)(*args)[indexer]
    result = q.simplify()
    expected = getattr(xdf[indexer], func)(*args)
    assert result._name == expected._name

    assert_eq(q, getattr(bdf, func)(*args)[indexer])

    q = getattr(xdf, func)(*args)[["x", "y"]]
    result = q.simplify()
    expected = getattr(xdf, func)(*args)
    assert result._name == expected._name


def test_sample(xdf):
    result = xdf.sample(frac=0.5)

    assert_eq(result, result)

    result = xdf.sample(frac=0.5, random_state=1234)
    expected = xdf.sample(frac=0.5, random_state=1234)
    assert_eq(result, expected)


def test_align(xdf, bdf, backend):
    if backend == "cudf":
        pytest.skip(reason="align not supported by cudf")
    result_1, result_2 = xdf.align(xdf)
    pdf_result_1, pdf_result_2 = bdf.align(bdf)
    assert_eq(result_1, pdf_result_1)
    assert_eq(result_2, pdf_result_2)

    result_1, result_2 = xdf.x.align(xdf.x)
    pdf_result_1, pdf_result_2 = bdf.x.align(bdf.x)
    assert_eq(result_1, pdf_result_1)
    assert_eq(result_2, pdf_result_2)


def test_align_different_partitions():
    pdf = pd.DataFrame({"a": [11, 12, 31, 1, 2, 3], "b": [1, 2, 3, 4, 5, 6]})
    df = from_pandas(pdf, npartitions=2)
    pdf2 = pd.DataFrame(
        {"a": [11, 12, 31, 1, 2, 3], "b": [1, 2, 3, 4, 5, 6]},
        index=[-2, -1, 0, 1, 2, 3],
    )
    df2 = from_pandas(pdf2, npartitions=2)
    result_1, result_2 = df.align(df2)
    pdf_result_1, pdf_result_2 = pdf.align(pdf2)
    assert_eq(result_1, pdf_result_1)
    assert_eq(result_2, pdf_result_2)


def test_align_unknown_partitions_same_root():
    pdf = pd.DataFrame({"a": 1}, index=[3, 2, 1])
    df = from_pandas(pdf, npartitions=2, sort=False)
    result_1, result_2 = df.align(df)
    pdf_result_1, pdf_result_2 = pdf.align(pdf)
    assert_eq(result_1, pdf_result_1)
    assert_eq(result_2, pdf_result_2)


def test_unknown_partitions_different_root():
    pdf = pd.DataFrame({"a": 1}, index=[3, 2, 1])
    df = from_pandas(pdf, npartitions=2, sort=False)
    pdf2 = pd.DataFrame({"a": 1}, index=[4, 3, 2, 1])
    df2 = from_pandas(pdf2, npartitions=2, sort=False)
    with pytest.raises(ValueError, match="Not all divisions"):
        df.align(df2)


def test_nunique_approx(xdf, backend):
    if backend == "cudf":
        pytest.xfail(reason="compute_hll_array doesn't work for cudf")
    result = xdf.nunique_approx().compute()
    assert 99 < result < 101


def test_assign_simplify(bdf):
    df = from_pandas(bdf)
    df2 = from_pandas(bdf)
    df["new"] = df.x > 1
    result = df[["x", "new"]].simplify()
    expected = df2[["x"]].assign(new=df2.x > 1).simplify()
    assert result._name == expected._name

    bdf["new"] = bdf.x > 1
    assert_eq(bdf[["x", "new"]], result)


def test_assign_simplify_new_column_not_needed(bdf):
    df = from_pandas(bdf)
    df2 = from_pandas(bdf)
    df["new"] = df.x > 1
    result = df[["x"]].simplify()
    expected = df2[["x"]].simplify()
    assert result._name == expected._name

    bdf["new"] = bdf.x > 1
    assert_eq(result, bdf[["x"]])


def test_assign_simplify_series(bdf):
    df = from_pandas(bdf)
    df2 = from_pandas(bdf)
    df["new"] = df.x > 1
    result = df.new.simplify()
    expected = df2[[]].assign(new=df2.x > 1).new.simplify()
    assert result._name == expected._name


def test_assign_non_series_inputs(xdf, bdf, backend):
    if backend == "cudf":
        pytest.xfail(reason="assign function not supported by cudf")
    assert_eq(xdf.assign(a=lambda x: x.x * 2), bdf.assign(a=lambda x: x.x * 2))
    assert_eq(xdf.assign(a=2), bdf.assign(a=2))
    assert_eq(xdf.assign(a=xdf.x.sum()), bdf.assign(a=bdf.x.sum()))

    assert_eq(xdf.assign(a=lambda x: x.x * 2).y, bdf.assign(a=lambda x: x.x * 2).y)
    assert_eq(xdf.assign(a=lambda x: x.x * 2).a, bdf.assign(a=lambda x: x.x * 2).a)


def test_are_co_aligned(bdf, xdf):
    df2 = xdf.reset_index()
    assert are_co_aligned(xdf.expr, df2.expr)
    assert are_co_aligned(xdf.expr, df2.sum().expr)
    assert not are_co_aligned(xdf.expr, df2.repartition(npartitions=2).expr)

    assert are_co_aligned(xdf.expr, xdf.sum().expr)
    assert are_co_aligned((xdf + xdf.sum()).expr, xdf.sum().expr)

    bdf = bdf.assign(z=1)
    df3 = from_pandas(bdf, npartitions=10)
    assert not are_co_aligned(xdf.expr, df3.expr)
    assert are_co_aligned(xdf.expr, df3.sum().expr)

    merged = xdf.merge(df2)
    merged_first = merged.reset_index()
    merged_second = merged.rename(columns={"x": "a"})
    assert are_co_aligned(merged_first.expr, merged_second.expr)
    assert not are_co_aligned(merged_first.expr, xdf.expr)


def test_astype_categories(xdf, backend):
    if backend == "cudf":
        pytest.xfail(reason="TODO")
    result = xdf.astype("category")
    assert_eq(result.x._meta.cat.categories, pd.Index([UNKNOWN_CATEGORIES]))
    assert_eq(result.y._meta.cat.categories, pd.Index([UNKNOWN_CATEGORIES]))
