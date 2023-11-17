import functools
from collections import namedtuple

import numpy as np
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs, _resample_series

from dask_expr._expr import Blockwise, Expr

BlockwiseDep = namedtuple(typename="BlockwiseDep", field_names=["iterable"])


class ResampleReduction(Expr):
    _parameters = [
        "frame",
        "rule",
        "how",
        "kwargs",
        "fill_value",
        "how_args",
        "how_kwargs",
    ]
    _defaults = {
        "closed": None,
        "label": None,
        "fill_value": np.nan,
        "kwargs": None,
        "how_args": (),
        "how_kwargs": None,
    }

    @functools.cached_property
    def _meta(self):
        return self.frame._meta.resample(self.rule, **self.kwargs)

    @functools.cached_property
    def kwargs(self):
        return {} if self.operand("kwargs") is None else self.operand("kwargs")

    @functools.cached_property
    def _resample_divisions(self):
        return _resample_bin_and_out_divs(self.obj.divisions, self.rule, **self.kwargs)

    def _lower(self):
        partitioned = self.obj.repartition(divisions=self._resample_divisions[0])
        output_divisions = self._resample_divisions[1]
        return ResampleAggregation(
            partitioned,
            BlockwiseDep(output_divisions[:-1]),
            BlockwiseDep(output_divisions[1:]),
            BlockwiseDep(["left"] * (len(output_divisions[1:]) - 1) + [None]),
            self.rule,
            self.kwargs,
            self.how,
            self.fill_value,
            list(self.how_args),
            self.kwargs,
        )


class ResampleAggregation(Blockwise):
    _parameters = [
        "frame",
        "divisions_left",
        "divisions_right",
        "closed",
        "rule",
        "kwargs",
        "how",
        "fill_value",
        "how_args",
        "how_kwargs",
    ]
    operation = _resample_series

    def _blockwise_arg(self, arg, i):
        if isinstance(arg, BlockwiseDep):
            return arg.iterable[i]
        return super()._blockwise_arg(arg, i)
