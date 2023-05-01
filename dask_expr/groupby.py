import functools

from dask.dataframe.core import _concat

# from dask.dataframe.groupby import _apply_chunk, _groupby_aggregate
from dask.dataframe.groupby import _determine_levels, _groupby_raise_unaligned
from dask.utils import M

from dask_expr.collection import new_collection
from dask_expr.expr import Expr
from dask_expr.reductions import ApplyConcatApply

###
### Groupby Expression API
###


class GroupBy(Expr):
    """Intermediate Groupby expresssion

    This is an abstract class in the sense that it
    cannot generate a task graph until it is converted
    to a scalar, series, or dataframe-like expression.

    Parameters
    ----------
    obj: Expr
        Dataframe- or series-like expression to group
    by: str, list or Series
        The key for grouping
    sort: bool
        Whether the output aggregation should have sorted keys.
    **options: dict
        Other groupby options to pass through to backend.
    """

    _parameters = [
        "obj",
        "by",
        "sort",
        "options",
    ]

    def _layer(self):
        raise NotImplementedError(
            f"{self} is abstract! Please use the Grouby API to "
            f"convert to a scalar, series, or dataframe-like "
            f"object before computing."
        )

    def _single_agg(
        self,
        func,
        aggfunc=None,
        chunk_kwargs=None,
        aggregate_kwargs=None,
        split_out=1,
        split_every=16,
    ):
        """Aggregation with a single function/aggfunc"""
        if aggfunc is None:
            aggfunc = func

        if chunk_kwargs is None:
            chunk_kwargs = {}

        if aggregate_kwargs is None:
            aggregate_kwargs = {}

        frame = self.obj
        by = self.by if isinstance(self.by, (list, tuple)) else [self.by]
        levels = _determine_levels(by)

        return SingleAgg(
            frame,
            by,
            func,
            chunk_kwargs,
            aggfunc,
            levels,
            aggregate_kwargs,
            self.options,
            split_out,
            split_every,
        )

    def count(self, split_out=1):
        return self._single_agg(
            func=M.count,
            aggfunc=M.sum,
            split_out=split_out,
        )


class SingleAgg(ApplyConcatApply):
    _parameters = [
        "frame",
        "by",
        "chunk",
        "chunk_kwargs",
        "aggregate",
        "levels",
        "aggregate_kwargs",
        "groupby_kwargs",
        "split_out",
        "split_every",
    ]

    @staticmethod
    def _apply_chunk(df, **kwargs):
        func = kwargs.pop("chunk")
        by = kwargs.pop("by")
        groupby_kwargs = kwargs.pop("groupby_kwargs")
        g = _groupby_raise_unaligned(df, by=by, **groupby_kwargs)
        return func(g, **kwargs)

    @staticmethod
    def _groupby_aggregate(dfs, **kwargs):
        aggfunc = kwargs.pop("aggfunc")
        levels = kwargs.pop("levels")
        groupby_kwargs = kwargs.pop("groupby_kwargs")
        grouped = _concat(dfs).groupby(level=levels, **groupby_kwargs)
        return aggfunc(grouped, **kwargs)

    @functools.cached_property
    def chunk(self):
        return functools.partial(
            self._apply_chunk,
            by=self.operand("by"),
            chunk=self.operand("chunk"),
            groupby_kwargs=self.operand("groupby_kwargs"),
        )

    @functools.cached_property
    def aggregate(self):
        return functools.partial(
            self._groupby_aggregate,
            aggfunc=self.operand("aggregate"),
            levels=self.operand("levels"),
            groupby_kwargs=self.operand("groupby_kwargs"),
        )

    @property
    def split_every(self):
        return self.operand("split_every")

    @property
    def chunk_kwargs(self):
        return self.operand("chunk_kwargs")

    @property
    def aggregate_kwargs(self):
        return self.operand("aggregate_kwargs")


###
### Collection Groupby API
###


class CollectionGroupBy:
    """Abstract Groupby-expression container"""

    def __init__(
        self,
        obj,
        by,
        sort=True,
        **options,
    ):
        for key in by if isinstance(by, (tuple, list)) else [by]:
            if not isinstance(key, (str, int)):
                raise NotImplementedError("Can only group on column names (for now).")

        self._expr = GroupBy(obj.expr, by, sort, options)

    @property
    def expr(self):
        return self._expr

    def count(self, split_out=1):
        return new_collection(self.expr.count(split_out=split_out))
