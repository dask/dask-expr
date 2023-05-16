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
    def assume(self, parent: Expr) -> Statistics | None:
        """Statistics that a "parent" Expr may assume

        A return value of `None` means that `type(Expr)` is
        not eligable to assume these kind of statistics.
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


@PartitionStatistics.assume.register
def _partitionstatistics_partitions(self, parent: Partitions):
    # A `Partitions` expression may assume statistics
    # from the selected partitions
    return type(self)(
        type(self.data)(
            part for i, part in enumerate(self.data) if i in parent.partitions
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


@RowCountStatistics.assume.register
def _rowcount_elemwise(self, parent: Elemwise):
    # All Element-wise operations may assume
    # row-count statistics
    return self
