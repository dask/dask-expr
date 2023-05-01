import pandas as pd
import pytest
from dask.dataframe.utils import assert_eq

from dask_expr import from_pandas


@pytest.mark.parametrize("split_out", [1])
def test_groupby_count(split_out):
    pdf = pd.DataFrame({"x": list(range(10)) * 10, "y": range(100)})
    df = from_pandas(pdf, npartitions=4)

    g = df.groupby("x")
    agg = g.count(split_out=split_out)

    expect = pdf.groupby("x").count()
    assert_eq(agg, expect)
