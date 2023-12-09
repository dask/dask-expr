import numpy as np
import pytest
from dask.array.utils import assert_eq

import dask_expr.array as da


def test_basic():
    x = np.random.random((10, 10))
    xx = da.from_array(x, chunks=(4, 4))
    xx._meta
    xx.chunks
    repr(xx)

    assert_eq(x, xx)


def test_rechunk():
    a = np.random.random((10, 10))
    b = da.from_array(a, chunks=(4, 4))
    c = b.rechunk()
    assert c.npartitions == 1
    assert_eq(b, c)

    d = b.rechunk((3, 3))
    assert d.npartitions == 16
    assert_eq(d, a)


def test_rechunk_optimize():
    a = np.random.random((10, 10))
    b = da.from_array(a, chunks=(4, 4))

    c = b.rechunk((2, 5)).rechunk((5, 2))
    d = b.rechunk((5, 2))

    assert c.optimize()._name == d.optimize()._name


def test_rechunk_blockwise_optimize():
    a = np.random.random((10, 10))
    b = da.from_array(a, chunks=(4, 4))

    result = (da.from_array(a, chunks=(4, 4)) + 1).rechunk((5, 5))
    expected = da.from_array(a, chunks=(5, 5)) + 1
    assert result.optimize()._name == expected.optimize()._name

    a = np.random.random((10,))
    aa = da.from_array(a)
    b = np.random.random((10, 10))
    bb = da.from_array(b)

    c = (aa + bb).rechunk((5, 2))
    result = c.optimize()
    expected = da.from_array(a, chunks=(2,)) + da.from_array(b, chunks=(5, 2))
    assert result._name == expected._name

    a = np.random.random((10, 1))
    aa = da.from_array(a)
    b = np.random.random((10, 10))
    bb = da.from_array(b)

    c = (aa + bb).rechunk((5, 2))
    result = c.optimize()

    expected = da.from_array(a, chunks=(5, 1)) + da.from_array(b, chunks=(5, 2))
    assert result._name == expected._name


def test_elemwise():
    a = np.random.random((10, 10))
    b = da.from_array(a, chunks=(4, 4))

    (b + 1).compute()
    assert_eq(a + 1, b + 1)
    assert_eq(a + 2 * a, b + 2 * b)

    x = np.random.random(10)
    y = da.from_array(x, chunks=(4,))

    assert_eq(a + x, b + y)


def test_transpose():
    a = np.random.random((10, 20))
    b = da.from_array(a, chunks=(2, 5))

    assert_eq(a.T, b.T)

    a = np.random.random((10, 1))
    b = da.from_array(a, chunks=(5, 1))
    assert_eq(a.T + a, b.T + b)
    assert_eq(a + a.T, b + b.T)

    assert b.T.T.optimize()._name == b.optimize()._name


def test_slicing():
    a = np.random.random((10, 20))
    b = da.from_array(a, chunks=(2, 5))

    assert_eq(a[:], b[:])
    assert_eq(a[::2], b[::2])
    assert_eq(a[1, :5], b[1, :5])
    assert_eq(a[None, ..., ::5], b[None, ..., ::5])
    assert_eq(a[3], b[3])


def test_slicing_optimization():
    a = np.random.random((10, 20))
    b = da.from_array(a, chunks=(2, 5))

    assert b[:].optimize()._name == b._name
    assert b[5:, 4][::2].optimize()._name == b[5::2, 4].optimize()._name

    assert (b + 1)[:5].optimize()._name == (b[:5] + 1)._name


@pytest.mark.xfail(reason="Blockwise specifies too much about dimension")
def test_slicing_optimization_change_dimensionality():
    a = np.random.random((10, 20))
    b = da.from_array(a, chunks=(2, 5))
    assert (b + 1)[5].optimize()._name == (b[5] + 1)._name


def test_xarray():
    import xarray as xr

    a = np.random.random((10, 20))
    b = da.from_array(a)

    x = (xr.DataArray(b, dims=["x", "y"]) + 1).chunk(x=2)

    assert x.data.optimize()._name == (da.from_array(a, chunks={0: 2}) + 1)._name