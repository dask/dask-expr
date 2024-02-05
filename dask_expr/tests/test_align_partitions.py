from collections import OrderedDict

import pytest

from dask_expr import from_pandas
from dask_expr._expr import OpAlignPartitions
from dask_expr._repartition import RepartitionDivisions
from dask_expr._shuffle import divisions_lru
from dask_expr.tests._util import _backend_library, assert_eq

# Set DataFrame backend for this module
pd = _backend_library()


@pytest.fixture
def pdf():
    pdf = pd.DataFrame({"x": range(100)})
    pdf["y"] = pdf.x // 7  # Not unique; duplicates span different partitions
    yield pdf


@pytest.fixture
def df(pdf):
    yield from_pandas(pdf, npartitions=10)


@pytest.mark.parametrize("op", ["__add__", "add"])
def test_broadcasting_scalar(pdf, df, op):
    df2 = from_pandas(pdf, npartitions=2)
    result = getattr(df, op)(df2.x.sum())
    assert_eq(result, pdf + pdf.x.sum())
    assert len(list(result.expr.find_operations(OpAlignPartitions))) == 0

    divisions_lru.data = OrderedDict()
    result = getattr(df.set_index("x"), op)(df2.x.sum())
    # Make sure that we don't touch divisions
    assert len(divisions_lru.data) == 0
    assert_eq(result, pdf.set_index("x") + pdf.x.sum())
    assert len(list(result.expr.find_operations(OpAlignPartitions))) == 0

    if op == "__add__":
        # Can't avoid expensive alignment check, but don't touch divisions while figuring it out
        divisions_lru.data = OrderedDict()
        result = getattr(df.set_index("x"), op)(df2.set_index("x").sum())
        # Make sure that we don't touch divisions
        assert len(divisions_lru.data) == 0
        assert_eq(result, pdf.set_index("x") + pdf.set_index("x").sum())
        assert len(list(result.expr.find_operations(OpAlignPartitions))) > 0

        assert (
            len(
                list(
                    result.optimize(fuse=False).expr.find_operations(
                        RepartitionDivisions
                    )
                )
            )
            == 0
        )

    # Can't avoid alignment, but don't touch divisions while figuring it out
    divisions_lru.data = OrderedDict()
    result = getattr(df.set_index("x"), op)(df2.set_index("x"))
    # Make sure that we don't touch divisions
    assert len(divisions_lru.data) == 0
    assert_eq(result, pdf.set_index("x") + pdf.set_index("x"))
    assert len(list(result.expr.find_operations(OpAlignPartitions))) > 0

    assert (
        len(
            list(result.optimize(fuse=False).expr.find_operations(RepartitionDivisions))
        )
        > 0
    )
