import operator
from typing import Union

import dask.array as da
from dask.base import DaskMethodsMixin, named_schedulers
from dask.utils import cached_cumsum, cached_property
from toolz import reduce

from dask_expr import _core as core

T_IntOrNaN = Union[int, float]  # Should be Union[int, Literal[np.nan]]


class Array(core.Expr, DaskMethodsMixin):
    _cached_keys = None

    __dask_scheduler__ = staticmethod(
        named_schedulers.get("threads", named_schedulers["sync"])
    )
    __dask_optimize__ = staticmethod(lambda dsk, keys, **kwargs: dsk)

    def __dask_postcompute__(self):
        return da.core.finalize, ()

    def __dask_postpersist__(self):
        return FromGraph, (self._meta, self.chunks, self._name)

    def __array_ufunc__(self, numpy_ufunc, method, *inputs, **kwargs):
        raise NotImplementedError()

    @cached_property
    def shape(self) -> tuple[T_IntOrNaN, ...]:
        return tuple(cached_cumsum(c, initial_zero=True)[-1] for c in self.chunks)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def chunksize(self) -> tuple[T_IntOrNaN, ...]:
        return tuple(max(c) for c in self.chunks)

    @property
    def dtype(self):
        if isinstance(self._meta, tuple):
            dtype = self._meta[0].dtype
        else:
            dtype = self._meta.dtype
        return dtype

    def __dask_keys__(self):
        if self._cached_keys is not None:
            return self._cached_keys

        name, chunks, numblocks = self.name, self.chunks, self.numblocks

        def keys(*args):
            if not chunks:
                return [(name,)]
            ind = len(args)
            if ind + 1 == len(numblocks):
                result = [(name,) + args + (i,) for i in range(numblocks[ind])]
            else:
                result = [keys(*(args + (i,))) for i in range(numblocks[ind])]
            return result

        self._cached_keys = result = keys()
        return result

    @cached_property
    def numblocks(self):
        return tuple(map(len, self.chunks))

    @cached_property
    def npartitions(self):
        return reduce(operator.mul, self.numblocks, 1)

    @property
    def name(self):
        return self._name

    def __hash__(self):
        return hash(self._name)

    def optimize(self):
        return self.simplify()

    def rechunk(
        self,
        chunks="auto",
        threshold=None,
        block_size_limit=None,
        balance=False,
        method=None,
    ):
        from dask_expr.array.rechunk import Rechunk

        return Rechunk(self, chunks, threshold, block_size_limit, balance, method)


class FromArray(Array):
    _parameters = ["array", "chunks"]

    @property
    def _meta(self):
        return self.array[tuple(slice(0, 0) for _ in range(self.array.ndim))]

    def _layer(self):
        dsk = da.core.graph_from_arraylike(
            self.array, chunks=self.chunks, shape=self.array.shape, name=self._name
        )
        return dict(dsk)  # this comes as a legacy HLG for now

    def __str__(self):
        return "FromArray(...)"


class FromGraph(Array):
    _parameters = ["layer", "_meta", "chunks", "_name"]

    @property
    def _meta(self):
        return self.operand("_meta")

    @property
    def chunks(self):
        return self.operand("chunks")

    @property
    def _name(self):
        return self.operand("_name")

    def _layer(self):
        return dict(self.operand("layer"))


def from_array(x, chunks="auto"):
    chunks = da.core.normalize_chunks(chunks, x.shape, dtype=x.dtype)
    return FromArray(x, chunks)
