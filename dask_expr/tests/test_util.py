import pickle

import pytest

from dask_expr._util import RaiseAttributeError, _tokenize_deterministic


def _clear_function_cache():
    from dask.base import function_cache, function_cache_lock

    with function_cache_lock:
        function_cache.clear()


def tokenize(x, /, __ensure_deterministic=True, *args, **kwargs):
    _clear_function_cache()
    try:
        before = _tokenize_deterministic(x, *args, **kwargs)
        _clear_function_cache()
        if __ensure_deterministic:
            assert before == _tokenize_deterministic(x, *args, **kwargs)
            _clear_function_cache()
        try:
            after = _tokenize_deterministic(
                pickle.loads(pickle.dumps(x)), *args, **kwargs
            )
        except (AttributeError, pickle.PicklingError):
            # If we go down this path we're almost certainly guaranteed to fail
            # since cloudpickle dumps are not deterministic and the tokenization
            # is relying on that
            import cloudpickle

            after = _tokenize_deterministic(
                cloudpickle.loads(cloudpickle.dumps(x)), *args, **kwargs
            )

        assert before == after
        return before
    finally:
        _clear_function_cache()


def test_raises_attribute_error():
    class A:
        def x(self):
            ...

    class B(A):
        x = RaiseAttributeError()

    assert hasattr(A, "x")
    assert hasattr(A(), "x")
    assert not hasattr(B, "x")
    assert not hasattr(B(), "x")
    with pytest.raises(AttributeError, match="'B' object has no attribute 'x'"):
        B.x
    with pytest.raises(AttributeError, match="'B' object has no attribute 'x'"):
        B().x


def test_tokenize_lambda():
    func = lambda x: x + 1
    tokenize(func)


import pandas as pd

from dask_expr import from_pandas


def test_tokenize_deterministic():
    ddf = from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    tokenize(ddf)

    def identity(x):
        return x

    ddf2 = ddf.map_partitions(identity)
    tokenize(ddf2)

    def identity(x):
        return x + 1

    ddf3 = ddf.map_partitions(identity)

    from distributed.protocol.pickle import dumps, loads

    assert loads(dumps(ddf))._name == ddf._name

    assert loads(dumps(ddf2))._name == ddf2._name
    assert tokenize(ddf2) != tokenize(ddf3)
