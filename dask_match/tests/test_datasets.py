from dask.dataframe.utils import assert_eq

from dask_match.datasets import timeseries


def test_timeseries():
    df = timeseries(freq="360 s", start="2000-01-01", end="2000-01-02")
    assert_eq(df, df)


def test_optimization():
    dtypes = {"x": int, "y": float}
    df = timeseries(dtypes=dtypes, seed=123)
    expected = timeseries(dtypes=dtypes, _projection=["x"], seed=123)
    result = df[["x"]].optimize()
    assert expected._name == result._name

    expected = timeseries(dtypes=dtypes, _projection="x", seed=123)["x"]
    result = df["x"].optimize(fuse=False)
    assert expected._name == result._name


def test_column_projection_deterministic():
    df = timeseries(freq="1H", start="2000-01-01", end="2000-01-02", seed=123)
    result_id = df[["id"]].optimize()
    result_id_x = df[["id", "x"]].optimize()
    assert_eq(result_id["id"], result_id_x["id"])
