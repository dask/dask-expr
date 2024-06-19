from __future__ import annotations

import os

import pytest

from dask_expr import read_parquet
from dask_expr.tests._util import _backend_library, assert_eq

distributed = pytest.importorskip("distributed")

from distributed import Client, LocalCluster
from distributed.utils_test import client as c  # noqa F401
from distributed.utils_test import gen_cluster

import dask_expr as dx

pd = _backend_library()


@pytest.fixture(params=["arrow"])
def filesystem(request):
    return request.param


def _make_file(dir, df=None, filename="myfile.parquet", **kwargs):
    fn = os.path.join(str(dir), filename)
    if df is None:
        df = pd.DataFrame({c: range(10) for c in "abcde"})
    df.to_parquet(fn, **kwargs)
    return fn


def test_io_fusion_merge(tmpdir):
    pdf = pd.DataFrame({c: range(100) for c in "abcdefghij"})
    with LocalCluster(processes=False, n_workers=2) as cluster:
        with Client(cluster) as client:  # noqa: F841
            dx.from_pandas(pdf, 10).to_parquet(tmpdir)
            df = dx.read_parquet(tmpdir).merge(
                dx.read_parquet(tmpdir).add_suffix("_x"), left_on="a", right_on="a_x"
            )[["a_x", "b_x", "b"]]
            out = df.compute()
    pd.testing.assert_frame_equal(
        out.sort_values(by="a_x", ignore_index=True),
        pdf.merge(pdf.add_suffix("_x"), left_on="a", right_on="a_x")[
            ["a_x", "b_x", "b"]
        ],
    )


@pytest.mark.filterwarnings("error")
@gen_cluster(client=True)
async def test_parquet_distriuted(c, s, a, b, tmpdir, filesystem):
    pdf = pd.DataFrame({"x": [1, 4, 3, 2, 0, 5]})
    df = read_parquet(_make_file(tmpdir, df=pdf), filesystem=filesystem)
    assert_eq(await c.gather(c.compute(df.optimize())), pdf)


def test_pickle_size(tmpdir, filesystem):
    pdf = pd.DataFrame({"x": [1, 4, 3, 2, 0, 5]})
    [_make_file(tmpdir, df=pdf, filename=f"{x}.parquet") for x in range(10)]
    df = read_parquet(tmpdir, filesystem=filesystem)
    from distributed.protocol import dumps

    assert len(b"".join(dumps(df.optimize().dask))) <= 8300
