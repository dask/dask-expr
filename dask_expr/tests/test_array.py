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
