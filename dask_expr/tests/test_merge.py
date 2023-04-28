import pandas as pd
import pytest
from dask.dataframe.utils import assert_eq

from dask_expr import from_pandas


@pytest.mark.parametrize("opt", [True, False])
@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
@pytest.mark.parametrize("shuffle_backend", ["tasks", "disk"])
def test_merge(opt, how, shuffle_backend):
    # Make simple left & right dfs
    pdf1 = pd.DataFrame({"x": range(20), "y": range(20)})
    df1 = from_pandas(pdf1, 4)
    pdf2 = pd.DataFrame({"x": range(0, 20, 2), "z": range(10)})
    df2 = from_pandas(pdf2, 2)

    # Partition-wise merge with map_partitions
    df3 = df1.merge(df2, on="x", how=how, shuffle_backend=shuffle_backend)

    # Check result with/without fusion
    expect = pdf1.merge(pdf2, on="x", how=how)
    df3 = df3.optimize() if opt else df3
    assert_eq(df3, expect, check_index=False)


@pytest.mark.parametrize("opt", [True, False])
@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
@pytest.mark.parametrize("pass_name", [True, False])
@pytest.mark.parametrize("shuffle_backend", ["tasks", "disk"])
def test_merge_indexed(opt, how, pass_name, shuffle_backend):
    # Make simple left & right dfs
    pdf1 = pd.DataFrame({"x": range(20), "y": range(20)})
    df1 = from_pandas(pdf1, 4)
    pdf2 = pd.DataFrame({"x": range(0, 20, 2), "z": range(10)}).set_index("x")
    df2 = from_pandas(pdf2, 2)

    left_index = True
    right_index = False if pass_name else True
    right_on = "x" if pass_name else None
    df3 = df1.merge(
        df2,
        left_index=left_index,
        right_index=right_index,
        right_on=right_on,
        how=how,
        shuffle_backend=shuffle_backend,
    )

    # Check result with/without fusion
    expect = pdf1.merge(
        pdf2,
        left_index=left_index,
        right_index=right_index,
        right_on=right_on,
        how=how,
    )
    df3 = df3.optimize() if opt else df3
    assert_eq(df3, expect)


@pytest.mark.parametrize("opt", [True, False])
@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
def test_broadcast_merge(opt, how):
    # Make simple left & right dfs
    pdf1 = pd.DataFrame({"x": range(20), "y": range(20)})
    df1 = from_pandas(pdf1, 4)
    pdf2 = pd.DataFrame({"x": range(0, 20, 2), "z": range(10)})
    df2 = from_pandas(pdf2, 1)

    df3 = df1.merge(df2, on="x", how=how)

    # Check that we avoid the shuffle when allowed
    if how in ("left", "inner"):
        assert all(["Shuffle" not in str(op) for op in df3.simplify().operands[:2]])

    # Check result with/without fusion
    expect = pdf1.merge(pdf2, on="x", how=how)
    df3 = df3.optimize() if opt else df3
    assert_eq(df3, expect, check_index=False)


def test_merge_column_projection():
    # Make simple left & right dfs
    pdf1 = pd.DataFrame({"x": range(20), "y": range(20)})
    df1 = from_pandas(pdf1, 4)
    pdf2 = pd.DataFrame({"x": range(0, 20, 2), "z": range(10)})
    df2 = from_pandas(pdf2, 2)

    # Partition-wise merge with map_partitions
    df3 = df1.merge(df2, on="x")["z"].simplify()

    assert "y" not in df3.expr.operands[0].columns
