import functools

from dask_expr.expr import Expr, MapPartitions
from dask_expr.io import FromPandas
from dask_expr.shuffle import Shuffle


class Merge(Expr):
    """Abstract merge operation"""

    _parameters = [
        "left",
        "right",
        "how",
        "on",
        "left_on",
        "right_on",
        "left_index",
        "right_index",
        "suffixes",
        "indicator",
        "shuffle_backend",
    ]
    _defaults = {
        "how": "inner",
        "on": None,
        "left_on": None,
        "right_on": None,
        "left_index": False,
        "right_index": False,
        "suffixes": ("_x", "_y"),
        "indicator": False,
        "shuffle_backend": None,
    }

    def __str__(self):
        return f"Merge({self._name[-7:]})"

    @property
    def _kwargs(self):
        return {
            k: self.operand(k)
            for k in [
                "how",
                "on",
                "left_on",
                "right_on",
                "left_index",
                "right_index",
                "suffixes",
                "indicator",
            ]
        }

    @functools.cached_property
    def _meta(self):
        if isinstance(self.right, Expr):
            right = self.right._meta
        else:
            right = self.right.iloc[:0]
        return self.left._meta.merge(right, **self._kwargs)

    def _divisions(self):
        npartitions_right = (
            self.right.npartitions if isinstance(self.right, Expr) else 1
        )
        npartitions = max(self.left.npartitions, npartitions_right)
        return (None,) * (npartitions + 1)

    @staticmethod
    def _merge_partition(df, other, **kwargs):
        return df.merge(other, **kwargs)

    def _simplify_down(self):
        # TODO:
        #  1. Handle merge on indices with known divisions
        #  2. Handle broadcast merge
        #  3. Add/leverage partition statistics

        left = self.left
        left_on = self.left_on or self.on
        npartitions_left = self.left.npartitions

        right = (
            self.right if isinstance(self.right, Expr) else FromPandas(self.right, 1)
        )
        right_on = self.right_on or self.on
        npartitions_right = (
            self.right.npartitions if isinstance(self.right, Expr) else 1
        )

        npartitions_out = max(npartitions_left, npartitions_right)

        # Shuffle left & right
        left = Shuffle(
            left, left_on, npartitions_out=npartitions_out, backend=self.shuffle_backend
        )
        right = Shuffle(
            right,
            right_on,
            npartitions_out=npartitions_out,
            backend=self.shuffle_backend,
        )

        # Partition-wise merge
        return MapPartitions(
            left,
            self._merge_partition,
            self._meta,
            False,
            True,
            self._kwargs,
            right,
        )
