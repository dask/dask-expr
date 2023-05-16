from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any

from dask_expr.expr import Elemwise, Expr, Partitions


@dataclass(frozen=True)
class Statistics:
    """Abstract expression-statistics class

    See Also
    --------
    PartitionStatistics
    """

    data: Any

    @singledispatchmethod
    def inherit(self, child: Expr) -> Statistics | None:
        """New `Statistics` object that a "child" Expr mayinherit

        A return value of `None` means that `type(Expr)` is
        not eligable to inherit this kind of statistics.
        """
        return None


@dataclass(frozen=True)
class PartitionStatistics(Statistics):
    """Statistics containing a distinct value for every partition

    See Also
    --------
    RowCountStatistics
    """

    data: Iterable


@PartitionStatistics.inherit.register
def _partitionstatistics_partitions(self, child: Partitions):
    # A `Partitions` expression may inherit statistics
    # from the selected partitions
    return type(self)(
        type(self.data)(
            part for i, part in enumerate(self.data) if i in child.partitions
        )
    )


#
# PartitionStatistics sub-classes
#


@dataclass(frozen=True)
class RowCountStatistics(PartitionStatistics):
    """Tracks the row count of each partition"""

    def sum(self):
        """Return the total row-count of all partitions"""
        return sum(self.data)


@RowCountStatistics.inherit.register
def _rowcount_elemwise(self, child: Elemwise):
    # All Element-wise operations may inherit
    # row-count statistics "as is"
    return self
