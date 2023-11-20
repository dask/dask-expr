import pytest

from dask_expr import from_pandas
from dask_expr.tests._util import _backend_library, assert_eq

# Set DataFrame backend for this module
lib = _backend_library()


@pytest.fixture
def pdf():
    idx = lib.date_range("2000-01-01", periods=12, freq="T")
    pdf = lib.DataFrame({"foo": range(len(idx))}, index=idx)
    pdf["bar"] = 1
    yield pdf


@pytest.fixture
def df(pdf):
    yield from_pandas(pdf, npartitions=4)


@pytest.mark.parametrize(
    "api",
    [
        "count",
        "prod",
        "mean",
        "sum",
        "min",
        "max",
        "first",
        "last",
        "var",
        "std",
        "size",
        "nunique",
        "median",
    ],
)
def test_resample_apis(df, pdf, api):
    result = getattr(df.resample("2T"), api)()
    expected = getattr(pdf.resample("2T"), api)()
    assert_eq(result, expected)

    # No column output
    if api not in ("size",):
        result = getattr(df.resample("2T"), api)()["foo"]
        expected = getattr(pdf.resample("2T"), api)()["foo"]
        assert_eq(result, expected)

        q = result.simplify()
        eq = getattr(df["foo"].resample("2T"), api)().simplify()
        assert q._name == eq._name
