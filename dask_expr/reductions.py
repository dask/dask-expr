import functools

import pandas as pd
import toolz
from dask.dataframe.core import (
    _concat,
    is_dataframe_like,
    is_series_like,
    make_meta,
    meta_nonempty,
)
from dask.utils import M, apply

from dask_expr.expr import Blockwise, Elemwise, Expr, Projection


class Map(Blockwise):
    _parameters = ["frame"]
    chunk = None
    chunk_kwargs = {}

    @functools.cached_property
    def _meta(self):
        meta = meta_nonempty(self.frame._meta)
        chunk_kwargs = self.chunk_kwargs or {}
        return make_meta(self.chunk(meta, **chunk_kwargs))

    def _task(self, index: int):
        args = [self._blockwise_arg(self.frame, index)]
        chunk_kwargs = self.chunk_kwargs or {}
        return (apply, self.chunk, args, chunk_kwargs)

    def _divisions(self):
        return (None,) * (self.frame.npartitions + 1)


class Reduce(Expr):
    _parameters = ["frame"]
    aggregate = None
    combine = None
    aggregate_kwargs = {}
    combine_kwargs = {}
    split_every = 0
    split_out = 1

    @property
    def _meta(self):
        aggregate = self.aggregate
        combine = self.combine or aggregate
        combine_kwargs = self.combine_kwargs or {}
        aggregate_kwargs = self.aggregate_kwargs or {}
        meta = meta_nonempty(self.frame._meta)
        meta = combine([meta], **combine_kwargs)
        meta = aggregate([meta], **aggregate_kwargs)
        return make_meta(meta)

    def _divisions(self):
        return (None,) * (self.split_out + 1)

    def __dask_postcompute__(self):
        return toolz.first, ()

    def _layer(self):
        aggregate = self.aggregate
        combine = self.combine or aggregate
        combine_kwargs = self.combine_kwargs or {}
        aggregate_kwargs = self.aggregate_kwargs or {}

        # apply combine to batches of intermediate results
        j = 1
        d = {}
        keys = list(self.frame.__dask_keys__())
        while len(keys) > 1:
            new_keys = []
            for i, batch in enumerate(
                toolz.partition_all(self.split_every or len(keys), keys)
            ):
                batch = list(batch)
                if combine_kwargs:
                    d[self._name, j, i] = (apply, combine, [batch], self.combine_kwargs)
                else:
                    d[self._name, j, i] = (combine, batch)
                new_keys.append((self._name, j, i))
            j += 1
            keys = new_keys

        # apply aggregate to the final result
        d[self._name, 0] = (apply, aggregate, [keys], aggregate_kwargs)

        return d


class MapReduce(Expr):
    _parameters = ["frame"]
    split_every: int = 0
    map: Expr | None = None
    reduce: Expr | None = None

    @property
    def _meta(self):
        mapped = self.map(*self.operands)
        reduced = self.reduce(mapped, *self.operands[1:])
        return reduced._meta

    def _divisions(self):
        return (None, None)

    def _simplify_down(self):
        mapped = self.map(*self.operands)
        return self.reduce(mapped, *self.operands[1:])


class ReductionMap(Map):
    reduction_chunk = None

    @classmethod
    def chunk(cls, df, **kwargs):
        out = cls.reduction_chunk(df, **kwargs)
        # Return a dataframe so that the concatenated version is also a dataframe
        return out.to_frame().T if is_series_like(out) else out

    @property
    def _meta(self):
        meta = meta_nonempty(self.frame._meta)
        chunk_kwargs = self.chunk_kwargs or {}
        meta = self.chunk(meta, **chunk_kwargs)
        return make_meta(meta)


class ReductionReduce(Reduce):
    reduction_combine = None
    reduction_aggregate = None

    @property
    def _meta(self):
        meta = meta_nonempty(self.frame._meta)
        combine_kwargs = self.combine_kwargs or {}
        meta = self.combine([meta], **combine_kwargs)
        aggregate_kwargs = self.aggregate_kwargs or {}
        meta = self.aggregate([meta], **aggregate_kwargs)
        return make_meta(meta)

    @classmethod
    def combine(cls, inputs: list, **kwargs):
        func = cls.reduction_combine or cls.reduction_aggregate or cls.reduction_chunk
        df = _concat(inputs)
        out = func(df, **kwargs)
        # Return a dataframe so that the concatenated version is also a dataframe
        return out.to_frame().T if is_series_like(out) else out

    @classmethod
    def aggregate(cls, inputs, **kwargs):
        func = cls.reduction_aggregate or cls.reduction_chunk
        df = _concat(inputs)
        return func(df, **kwargs)


class Reduction(MapReduce):
    def __str__(self):
        params = {param: self.operand(param) for param in self._parameters[1:]}
        s = ", ".join(
            k + "=" + repr(v) for k, v in params.items() if v != self._defaults.get(k)
        )
        base = str(self.frame)
        if " " in base:
            base = "(" + base + ")"
        return f"{base}.{self.__class__.__name__.lower()}({s})"


##
## Templates
##


class MapReduceTemplate:
    _map_base = Map
    _reduce_base = Reduce

    @classmethod
    def make_cls(cls, base, name: str):
        class _cls(cls, base):
            pass

        _cls.__name__ = name
        return _cls

    @classmethod
    def make_map(cls, label: str = ""):
        name = (label or cls.__name__.split("_")[-1]) + "Map"
        return cls.make_cls(cls._map_base, name)

    @classmethod
    def make_reduce(cls, label: str = ""):
        name = (label or cls.__name__.split("_")[-1]) + "Reduce"
        return cls.make_cls(cls._reduce_base, name)


class ReductionTemplate(MapReduceTemplate):
    _map_base = ReductionMap
    _reduce_base = ReductionReduce


##
## Sum
##


class _Sum(ReductionTemplate):
    _parameters = ["frame", "skipna", "numeric_only", "min_count"]
    _defaults = {"skipna": True, "numeric_only": None, "min_count": 0}
    reduction_chunk = M.sum
    reduction_aggregate = M.sum

    @property
    def chunk_kwargs(self):
        return dict(
            skipna=self.skipna,
            numeric_only=self.numeric_only,
            min_count=self.min_count,
        )

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            return self.frame[parent.operand("columns")].sum(*self.operands[1:])


class Sum(_Sum, Reduction):
    map = _Sum.make_map()
    reduce = _Sum.make_reduce()


##
## Max
##


class _Max(ReductionTemplate):
    _parameters = ["frame", "skipna", "numeric_only"]
    _defaults = {"skipna": True, "numeric_only": None}
    reduction_chunk = M.max

    @property
    def chunk_kwargs(self):
        return dict(
            skipna=self.skipna,
            numeric_only=self.numeric_only,
        )

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            return self.frame[parent.operand("columns")].max(skipna=self.skipna)


class Max(_Max, Reduction):
    map = _Max.make_map()
    reduce = _Max.make_reduce()


##
## Len
##


class _Len(ReductionTemplate):
    _parameters = ["frame"]
    reduction_chunk = staticmethod(len)
    reduction_aggregate = M.sum


class Len(_Len, Reduction):
    map = _Len.make_map()
    reduce = _Len.make_reduce()

    def _simplify_down(self):
        if isinstance(self.frame, Elemwise):
            child = max(self.frame.dependencies(), key=lambda expr: expr.npartitions)
            return Len(child)


##
## Size
##


class _Size(ReductionTemplate):
    _parameters = ["frame"]
    reduction_chunk = staticmethod(lambda df: df.size)
    reduction_aggregate = sum


class Size(_Size, Reduction):
    map = _Size.make_map()
    reduce = _Size.make_reduce()

    def _simplify_down(self):
        if is_dataframe_like(self.frame) and len(self.frame.columns) > 1:
            return len(self.frame.columns) * Len(self.frame)
        else:
            return Len(self.frame)


##
## Mean
##


class Mean(Reduction):
    _parameters = ["frame", "skipna", "numeric_only"]
    _defaults = {"skipna": True, "numeric_only": None}

    @property
    def _meta(self):
        return (
            self.frame._meta.sum(skipna=self.skipna, numeric_only=self.numeric_only) / 2
        )

    def _simplify_down(self):
        return (
            self.frame.sum(skipna=self.skipna, numeric_only=self.numeric_only)
            / self.frame.count()
        )


##
## Count
##


class _Count(ReductionTemplate):
    _parameters = ["frame", "numeric_only"]
    _defaults = {"numeric_only": None}
    split_every = 16
    reduction_chunk = M.count

    @classmethod
    def reduction_aggregate(cls, df):
        return df.sum().astype("int64")


class Count(_Count, Reduction):
    map = _Count.make_map()
    reduce = _Count.make_reduce()


##
## Min
##


class _Min(_Max):
    reduction_chunk = M.min


class Min(_Min, Reduction):
    map = _Min.make_map()
    reduce = _Min.make_reduce()


##
## Mode
##


class _Mode(MapReduceTemplate):
    """

    Mode was a bit more complicated than class reductions,
    so we retreat back to MapReduce
    """

    _parameters = ["frame", "dropna"]
    _defaults = {"dropna": True}
    chunk = M.value_counts
    split_every = 16

    @classmethod
    def combine(cls, results: list[pd.Series]):
        df = _concat(results)
        out = df.groupby(df.index).sum()
        out.name = results[0].name
        return out

    @classmethod
    def aggregate(cls, results: list[pd.Series], dropna=None):
        [df] = results
        max = df.max(skipna=dropna)
        out = df[df == max].index.to_series().sort_values().reset_index(drop=True)
        out.name = results[0].name
        return out

    @property
    def chunk_kwargs(self):
        return {"dropna": self.dropna}

    @property
    def aggregate_kwargs(self):
        return {"dropna": self.dropna}


class Mode(_Mode, MapReduce):
    map = _Mode.make_map()
    reduce = _Mode.make_reduce()
