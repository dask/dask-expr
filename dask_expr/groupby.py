import functools

import numpy as np
from dask.dataframe.core import (
    _concat,
    is_dataframe_like,
    is_series_like,
    make_meta,
    meta_nonempty,
    no_default,
)
from dask.dataframe.groupby import (
    _agg_finalize,
    _apply_chunk,
    _build_agg_args,
    _determine_levels,
    _groupby_aggregate,
    _groupby_apply_funcs,
    _normalize_spec,
)
from dask.utils import M
from pandas.core.apply import reconstruct_func

from dask_expr.collection import DataFrame, Series, new_collection
from dask_expr.reductions import ApplyConcatApply

###
### Groupby-aggregation expressions
###


class GroupbyAggregation(ApplyConcatApply):
    """General groupby aggregation

    This is an abstract class in the sense that it
    cannot generate a task graph until it is converted
    to a scalar, series, or dataframe-like expression.

    Sub-classes must implement the following methods:

    -   `groupby_chunk`: Applied to each group within
        the `chunk` method of `ApplyConcatApply`
    -   `groupby_aggregate`: Applied to each group within
        the `aggregate` method of `ApplyConcatApply`

    Parameters
    ----------
    frame: Expr
        Dataframe- or series-like expression to group.
    by: str, list or Series
        The key for grouping
    observed:
        Passed through to dataframe backend.
    dropna:
        Whether rows with NA values should be dropped.
    chunk_kwargs:
        Key-word arguments to pass to `groupby_chunk`.
    aggregate_kwargs:
        Key-word arguments to pass to `aggregate_chunk`.
    """

    _parameters = [
        "frame",
        "by",
        "observed",
        "dropna",
        "chunk_kwargs",
        "aggregate_kwargs",
    ]
    _defaults = {
        "observed": None,
        "dropna": None,
        "chunk_kwargs": None,
        "aggregate_kwargs": None,
    }

    groupby_chunk = None
    groupby_aggregate = None

    @classmethod
    def chunk(cls, df, by=None, **kwargs):
        return _apply_chunk(df, *by, **kwargs)

    @classmethod
    def aggregate(cls, inputs, **kwargs):
        return _groupby_aggregate(_concat(inputs), **kwargs)

    @property
    def dropna(self) -> dict:
        dropna = self.operand("dropna")
        if dropna is not None:
            return {"dropna": dropna}
        return {}

    @property
    def observed(self) -> dict:
        observed = self.operand("observed")
        if observed is not None:
            return {"observed": observed}
        return {}

    @property
    def chunk_kwargs(self) -> dict:
        chunk_kwargs = self.operand("chunk_kwargs") or {}
        meta = make_meta(
            self.groupby_chunk(
                meta_nonempty(self.frame._meta),
                **chunk_kwargs,
            )
        )
        columns = meta.name if is_series_like(meta) else meta.columns
        return {
            "chunk": self.groupby_chunk,
            "columns": columns,
            "by": self.by,
            **self.observed,
            **self.dropna,
            **chunk_kwargs,
        }

    @property
    def aggregate_kwargs(self) -> dict:
        groupby_aggregate = self.groupby_aggregate or self.groupby_chunk
        return {
            "aggfunc": groupby_aggregate,
            "levels": _determine_levels(self.by),
            **self.observed,
            **self.dropna,
            **(self.operand("aggregate_kwargs") or {}),
        }

    def _divisions(self):
        return (None, None)


class CustomAggregation(ApplyConcatApply):
    _parameters = [
        "frame",
        "by",
        "chunk_funcs",
        "aggregate_funcs",
        "finalizers",
        "observed",
        "dropna",
    ]
    _defaults = {
        "observed": None,
        "dropna": None,
    }

    @classmethod
    def chunk(cls, df, by=None, **kwargs):
        return _groupby_apply_funcs(df, *by, **kwargs)

    @classmethod
    def combine(cls, inputs, **kwargs):
        return _groupby_apply_funcs(_concat(inputs), **kwargs)

    @classmethod
    def aggregate(cls, inputs, **kwargs):
        return _agg_finalize(_concat(inputs), **kwargs)

    @property
    def levels(self):
        if isinstance(self.by, (tuple, list)) and len(self.by) > 1:
            levels = list(range(len(self.by)))
        else:
            levels = 0
        return levels

    @property
    def dropna(self) -> dict:
        dropna = self.operand("dropna")
        if dropna is not None:
            return {"dropna": dropna}
        return {}

    @property
    def observed(self) -> dict:
        observed = self.operand("observed")
        if observed is not None:
            return {"observed": observed}
        return {}

    @property
    def chunk_kwargs(self) -> dict:
        return {
            "funcs": self.chunk_funcs,
            "sort": False,
            "by": self.by,
            **self.observed,
            **self.dropna,
        }

    @property
    def combine_kwargs(self) -> dict:
        return {
            "funcs": self.aggregate_funcs,
            "level": self.levels,
            "sort": False,
            **self.observed,
            **self.dropna,
        }

    @property
    def aggregate_kwargs(self) -> dict:
        return {
            "aggregate_funcs": self.aggregate_funcs,
            "finalize_funcs": self.finalizers,
            "level": self.levels,
            **self.observed,
            **self.dropna,
        }


class Sum(GroupbyAggregation):
    groupby_chunk = M.sum


class Min(GroupbyAggregation):
    groupby_chunk = M.min


class Max(GroupbyAggregation):
    groupby_chunk = M.max


class Count(GroupbyAggregation):
    groupby_chunk = M.count
    groupby_aggregate = M.sum


class Mean(GroupbyAggregation):
    @functools.cached_property
    def _meta(self):
        return self.simplify()._meta

    def _simplify_down(self):
        s = Sum(*self.operands)
        # Drop chunk/aggregate_kwargs for count
        c = Count(*self.operands[:4])
        if is_dataframe_like(s._meta):
            c = c[s.columns]
        return s / c


###
### Groupby Collection API
###


class GroupBy:
    """Collection container for groupby aggregations

    The purpose of this class is to expose an API similar
    to Pandas' `Groupby` for dask-expr collections.

    See Also
    --------
    GroupbyAggregation
    """

    def __init__(
        self,
        obj,
        by,
        sort=None,
        observed=None,
        dropna=None,
    ):
        self.by = by if isinstance(by, (tuple, list)) else [by]
        self.obj = obj
        self.sort = sort
        self.observed = observed
        self.dropna = dropna

        for key in self.by:
            if not isinstance(key, (str, int)):
                raise NotImplementedError("Can only group on column names (for now).")

        if self.sort:
            raise NotImplementedError("sort=True not yet supported.")

    def _numeric_only_kwargs(self, numeric_only):
        kwargs = {} if numeric_only is no_default else {"numeric_only": numeric_only}
        return {"chunk_kwargs": kwargs, "aggregate_kwargs": kwargs}

    def _single_agg(
        self, expr_cls, split_out=1, chunk_kwargs=None, aggregate_kwargs=None
    ):
        if split_out > 1:
            raise NotImplementedError("split_out>1 not yet supported")
        return new_collection(
            expr_cls(
                self.obj.expr,
                self.by,
                self.observed,
                self.dropna,
                chunk_kwargs=chunk_kwargs,
                aggregate_kwargs=aggregate_kwargs,
            )
        )

    def count(self, **kwargs):
        return self._single_agg(Count, **kwargs)

    def sum(self, numeric_only=no_default, **kwargs):
        numeric_kwargs = self._numeric_only_kwargs(numeric_only)
        return self._single_agg(Sum, **kwargs, **numeric_kwargs)

    def mean(self, numeric_only=no_default, **kwargs):
        numeric_kwargs = self._numeric_only_kwargs(numeric_only)
        return self._single_agg(Mean, **kwargs, **numeric_kwargs)

    def min(self, numeric_only=no_default, **kwargs):
        numeric_kwargs = self._numeric_only_kwargs(numeric_only)
        return self._single_agg(Min, **kwargs, **numeric_kwargs)

    def max(self, numeric_only=no_default, **kwargs):
        numeric_kwargs = self._numeric_only_kwargs(numeric_only)
        return self._single_agg(Max, **kwargs, **numeric_kwargs)

    def aggregate(self, arg=None, split_every=None, split_out=1, **kwargs):
        relabeling = None
        if arg is None:
            relabeling, arg, columns, order = reconstruct_func(arg, **kwargs)
        if relabeling:
            raise NotImplementedError()
        # TODO: Handle "SeriesGroupBy" case

        if isinstance(self.obj, DataFrame):
            if isinstance(self.by, tuple) or np.isscalar(self.by):
                group_columns = {self.by}

            elif isinstance(self.by, list):
                group_columns = {
                    i for i in self.by if isinstance(i, tuple) or np.isscalar(i)
                }

            else:
                group_columns = set()

            # NOTE: this step relies on the by normalization to replace
            #       series with their name.
            non_group_columns = [
                col for col in self.obj.columns if col not in group_columns
            ]
            # TODO: need to colver "slice" case

            spec = _normalize_spec(arg, non_group_columns)

        elif isinstance(self.obj, Series):
            if isinstance(arg, (list, tuple, dict)):
                # implementation detail: if self.obj is a series, a pseudo column
                # None is used to denote the series itself. This pseudo column is
                # removed from the result columns before passing the spec along.
                spec = _normalize_spec({None: arg}, [])
                spec = [
                    (result_column, func, input_column)
                    for ((_, result_column), func, input_column) in spec
                ]

            else:
                spec = _normalize_spec({None: arg}, [])
                spec = [
                    (self.obj.name, func, input_column)
                    for (_, func, input_column) in spec
                ]

        else:
            raise ValueError(f"aggregate on unknown object {self.obj}")

        chunk_funcs, aggregate_funcs, finalizers = _build_agg_args(spec)

        has_median = any(s[1] in ("median", np.median) for s in spec)
        if has_median:
            raise NotImplementedError()

        return new_collection(
            CustomAggregation(
                self.obj.expr,
                self.by,
                chunk_funcs,
                aggregate_funcs,
                finalizers,
                self.observed,
                self.dropna,
            )
        )
