from __future__ import annotations

import functools
from collections import OrderedDict, UserDict
from collections.abc import Hashable, Sequence
from types import LambdaType
from typing import Any, TypeVar, cast

import dask
from dask import config
from dask.base import normalize_token, tokenize
from dask.utils import Dispatch
from packaging.version import Version

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

DASK_VERSION = Version(dask.__version__)
DASK_GT_20231000 = DASK_VERSION > Version("2023.10.0")


def _convert_to_list(column) -> list | None:
    if column is None or isinstance(column, list):
        pass
    elif isinstance(column, tuple):
        column = list(column)
    elif hasattr(column, "dtype"):
        column = column.tolist()
    else:
        column = [column]
    return column


def is_scalar(x):
    # np.isscalar does not work for some pandas scalars, for example pd.NA
    if isinstance(x, Sequence) and not isinstance(x, str) or hasattr(x, "dtype"):
        return False
    if isinstance(x, (str, int)):
        return True

    from dask_expr._expr import Expr

    return not isinstance(x, Expr)


@normalize_token.register(LambdaType)
def _normalize_lambda(func):
    return str(func)


def _tokenize_deterministic(*args, **kwargs) -> str:
    # Utility to be strict about deterministic tokens
    with config.set({"tokenize.ensure-deterministic": True}):
        return tokenize(*args, **kwargs)


def _tokenize_partial(expr, ignore: list | None = None) -> str:
    # Helper function to "tokenize" the operands
    # that are not in the `ignore` list
    ignore = ignore or []
    return _tokenize_deterministic(
        *[
            op
            for i, op in enumerate(expr.operands)
            if i >= len(expr._parameters) or expr._parameters[i] not in ignore
        ]
    )


class LRU(UserDict[K, V]):
    """Limited size mapping, evicting the least recently looked-up key when full"""

    def __init__(self, maxsize: float) -> None:
        super().__init__()
        self.data = OrderedDict()
        self.maxsize = maxsize

    def __getitem__(self, key: K) -> V:
        value = super().__getitem__(key)
        cast(OrderedDict, self.data).move_to_end(key)
        return value

    def __setitem__(self, key: K, value: V) -> None:
        if len(self) >= self.maxsize:
            cast(OrderedDict, self.data).popitem(last=False)
        super().__setitem__(key, value)


class _BackendData:
    """Helper class to wrap backend data

    The primary purpose of this class is to provide
    caching outside the ``FromPandas`` class.
    """

    def __init__(self, data):
        self._data = data
        self._division_info = LRU(10)

    @functools.cached_property
    def _token(self):
        from dask_expr._util import _tokenize_deterministic

        return _tokenize_deterministic(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, key: str) -> Any:
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            # Return the underlying backend attribute
            return getattr(self._data, key)

    def __reduce__(self):
        return type(self), (self._data,)


@normalize_token.register(_BackendData)
def normalize_data_wrapper(data):
    return data._token


##
## External class-dispatching utilities
##

__ext_dispatch_classes = {}  # Track registered "external" dispatch classes


def _register_ext_dispatch(cls: type, meta_type: type, ext_cls: type | None = None):
    """Register a custom class for type-based dispatching"""

    def wrapper(ext_cls):
        if cls not in __ext_dispatch_classes:
            __ext_dispatch_classes[cls] = Dispatch(f"{cls.__qualname__}_dispatch")
        if isinstance(meta_type, tuple):
            for t in meta_type:
                __ext_dispatch_classes[cls].register(t, ext_cls)
        else:
            __ext_dispatch_classes[cls].register(meta_type, ext_cls)
        return ext_cls

    return wrapper(ext_cls) if ext_cls is not None else wrapper


def _get_ext_dispatch(cls: type, meta: Any) -> None | type:
    """Get the registered dispatch class if one exists"""
    try:
        return __ext_dispatch_classes[cls].dispatch(type(meta))
    except (KeyError, TypeError):
        # Return None by default
        return None
