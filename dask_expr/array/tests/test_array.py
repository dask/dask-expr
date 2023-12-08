import numpy as np
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


def test_elemwise():
    a = np.random.random((10, 10))
    b = da.from_array(a, chunks=(4, 4))

    (b + 1).compute()
    assert_eq(a + 1, b + 1)
    assert_eq(a + 2 * a, b + 2 * b)

    x = np.random.random(10)
    y = da.from_array(x, chunks=(4,))

    assert_eq(a + x, b + y)
