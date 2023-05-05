import functools

import numpy as np
from dask.base import normalize_token
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
    _groupby_aggregate,
    _groupby_apply_funcs,
    _normalize_spec,
)
from dask.utils import M

from dask_expr.collection import DataFrame, new_collection
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
    def levels(self):
        if isinstance(self.by, (tuple, list)) and len(self.by) > 1:
            levels = list(range(len(self.by)))
        else:
            levels = 0
        return levels

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
            "levels": self.levels,
            **self.observed,
            **self.dropna,
            **(self.operand("aggregate_kwargs") or {}),
        }

    def _divisions(self):
        return (None, None)


class AggregationSpec:
    """Utility class to wrap a raw groupby-aggregation spec"""

    def __init__(self, spec):
        self.spec = spec

    def __str__(self):
        return f"{self.spec}"

    def __repr__(self):
        return str(self)

    def build(self, frame, by):
        group_columns = set(by)
        non_group_columns = [col for col in frame.columns if col not in group_columns]
        spec = _normalize_spec(self.spec, non_group_columns)

        # Median not supported yet
        has_median = any(s[1] in ("median", np.median) for s in spec)
        if has_median:
            raise NotImplementedError("median not yet supported")

        chunk_funcs, aggregate_funcs, finalizers = _build_agg_args(spec)
        return {
            "chunk_funcs": chunk_funcs,
            "aggregate_funcs": aggregate_funcs,
            "finalizers": finalizers,
        }


@normalize_token.register(AggregationSpec)
def normalize_expression(spec):
    return spec.spec


class CustomAggregation(GroupbyAggregation):
    _parameters = [
        "frame",
        "by",
        "aggs",
        "observed",
        "dropna",
        "split_every",
    ]
    _defaults = {
        "observed": None,
        "dropna": None,
        "split_every": 8,
    }

    @property
    def split_every(self):
        return self.operand("split_every")

    @functools.cached_property
    def spec(self):
        return self.aggs.build(self.frame, self.by)

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
    def chunk_kwargs(self) -> dict:
        return {
            "funcs": self.spec["chunk_funcs"],
            "sort": False,
            "by": self.by,
            **self.observed,
            **self.dropna,
        }

    @property
    def combine_kwargs(self) -> dict:
        return {
            "funcs": self.spec["aggregate_funcs"],
            "level": self.levels,
            "sort": False,
            **self.observed,
            **self.dropna,
        }

    @property
    def aggregate_kwargs(self) -> dict:
        return {
            "aggregate_funcs": self.spec["aggregate_funcs"],
            "finalize_funcs": self.spec["finalizers"],
            "level": self.levels,
            **self.observed,
            **self.dropna,
        }

    def _simplify_down(self):
        # Use agg-spec information to add column projection
        column_projection = None
        spec = self.aggs.spec
        if isinstance(spec, dict):
            column_projection = (
                set(self.by).union(spec.keys()).intersection(self.frame.columns)
            )
        if column_projection and column_projection < set(self.frame.columns):
            return type(self)(self.frame[list(column_projection)], *self.operands[1:])


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
        self.by = [by] if np.isscalar(by) else list(by)
        self.obj = obj
        self.sort = sort
        self.observed = observed
        self.dropna = dropna

        if not isinstance(self.obj, DataFrame):
            raise NotImplementedError(
                "groupby only supports DataFrame collections for now."
            )

        for key in self.by:
            if not (np.isscalar(key) and key in self.obj.columns):
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

    def aggregate(self, spec=None, split_every=8, split_out=1):
        if spec is None:
            raise NotImplementedError("spec=None not supported")

        if split_out > 1:
            raise NotImplementedError("split_out>1 not yet supported")

        return new_collection(
            CustomAggregation(
                self.obj.expr,
                self.by,
                AggregationSpec(spec),
                self.observed,
                self.dropna,
                split_every,
            )
        )

    def agg(self, *args, **kwargs):
        return self.aggregate(*args, **kwargs)
