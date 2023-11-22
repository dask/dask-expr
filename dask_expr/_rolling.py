import functools
from collections import namedtuple
from numbers import Integral

from dask.dataframe.dispatch import meta_nonempty

from dask_expr._collection import new_collection
from dask_expr._expr import Blockwise, Expr, Projection, make_meta

BlockwiseDep = namedtuple(typename="BlockwiseDep", field_names=["iterable"])


class RollingReduction(Expr):
    _parameters = [
        "frame",
        "window",
        "kwargs",
        "how_args",
        "how_kwargs",
    ]
    _defaults = {
        "kwargs": None,
        "how_args": (),
        "how_kwargs": None,
    }
    how = None

    @functools.cached_property
    def npartitions(self):
        return self.frame.npartitions

    def _divisions(self):
        return self.frame.divisions

    @functools.cached_property
    def _meta(self):
        rolling = meta_nonempty(self.frame._meta).rolling(self.window, **self.kwargs)
        meta = getattr(rolling, self.how)(*self.how_args, **self.how_kwargs or {})
        return make_meta(meta)

    @functools.cached_property
    def kwargs(self):
        return {} if self.operand("kwargs") is None else self.operand("kwargs")

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            return type(self)(self.frame[parent.operand("columns")], *self.operands[1:])

    @property
    def _has_single_partition(self):
        return (
            self.kwargs.get("axis") in (1, "columns")
            or (isinstance(self.window, Integral) and self.window <= 1)
            or self.frame.npartitions == 1
        )

    def _lower(self):
        return RollingAggregation(
            self.frame,
            self._has_single_partition,
            self.window,
            self.kwargs,
            self.how,
            list(self.how_args),
            self.how_kwargs,
        )


class RollingAggregation(Blockwise):
    _parameters = [
        "frame",
        "_has_single_partition",
        "window",
        "kwargs",
        "how",
        "how_args",
        "how_kwargs",
    ]

    @staticmethod
    def operation(
        frame, _has_single_partition, window, kwargs, how, how_args, how_kwargs
    ):
        if _has_single_partition:
            rolling = frame.rolling(window, **kwargs)
            return getattr(rolling, how)(*how_args, **(how_kwargs or {}))
        else:
            raise NotImplementedError("Not implemented for more than 1 partitions")

    @functools.cached_property
    def _meta(self):
        rolling = meta_nonempty(self.frame._meta).rolling(self.window, **self.kwargs)
        meta = getattr(rolling, self.how)(*self.how_args, **self.how_kwargs or {})
        return make_meta(meta)


class RollingCount(RollingReduction):
    how = "count"


class RollingSum(RollingReduction):
    how = "sum"


class RollingMean(RollingReduction):
    how = "mean"


class RollingMin(RollingReduction):
    how = "min"


class RollingMax(RollingReduction):
    how = "max"


class RollingVar(RollingReduction):
    how = "var"


class RollingStd(RollingReduction):
    how = "std"


class RollingMedian(RollingReduction):
    how = "median"


class RollingQuantile(RollingReduction):
    how = "quantile"


class RollingSkew(RollingReduction):
    how = "skew"


class RollingKurt(RollingReduction):
    how = "kurt"


class RollingAgg(RollingReduction):
    how = "agg"

    def _simplify_up(self, parent):
        # Disable optimization in `agg`; function may access other columns
        return


class Rolling:
    """Aggregate using one or more operations

    The purpose of this class is to expose an API similar
    to Pandas' `Rollingr` for dask-expr
    """

    def __init__(self, obj, window, **kwargs):
        if obj.divisions[0] is None:
            msg = (
                "Can only rolling dataframes with known divisions\n"
                "See https://docs.dask.org/en/latest/dataframe-design.html#partitions\n"
                "for more information."
            )
            raise ValueError(msg)
        self.obj = obj
        self.window = window
        self.kwargs = kwargs

    def _single_agg(self, expr_cls, how_args=(), how_kwargs=None):
        return new_collection(
            expr_cls(
                self.obj.expr,
                self.window,
                kwargs=self.kwargs,
                how_args=how_args,
                how_kwargs=how_kwargs,
            )
        )

    def count(self):
        return self._single_agg(RollingCount)

    def sum(self):
        return self._single_agg(RollingSum)

    def mean(self):
        return self._single_agg(RollingMean)

    def min(self):
        return self._single_agg(RollingMin)

    def max(self):
        return self._single_agg(RollingMax)

    def var(self):
        return self._single_agg(RollingVar)

    def std(self):
        return self._single_agg(RollingStd)

    def median(self):
        return self._single_agg(RollingMedian)

    def quantile(self, q):
        return self._single_agg(RollingQuantile, how_args=(q,))

    def skew(self):
        return self._single_agg(RollingSkew)

    def kurt(self):
        return self._single_agg(RollingKurt)

    def agg(self, func, *args, **kwargs):
        return self._single_agg(RollingAgg, how_args=(func, *args), how_kwargs=kwargs)