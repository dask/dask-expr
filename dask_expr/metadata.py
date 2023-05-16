from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any

from dask_expr.expr import Elemwise, Expr, Partitions


@dataclass(frozen=True)
class Metadata:
    """Abstract expression-metadata class

    See Also
    --------
    StaticMetadata
    PartitionMetadata
    """

    data: Any

    @singledispatchmethod
    def inherit(self, child: Expr) -> Metadata | None:
        """New `Metadata` object that a "child" Expr mayinherit

        A return value of `None` means that `type(Expr)` is
        not eligable to inherit this kind of metadata.
        """
        return None


@dataclass(frozen=True)
class StaticMetadata(Metadata):
    """A static metadata object

    This metadata is not partition-specific, and can be
    inherited by any child `Expr`.
    """

    def inherit(self, child: Expr) -> StaticMetadata:
        return self


@dataclass(frozen=True)
class PartitionMetadata(Metadata):
    """Metadata containing a distinct value for every partition

    See Also
    --------
    RowCountMetadata
    """

    data: Iterable


@PartitionMetadata.inherit.register
def _partitionmetadata_partitions(self, child: Partitions):
    # A `Partitions` expression may inherit metadata
    # from the selected partitions
    return type(self)(
        type(self.data)(
            part for i, part in enumerate(self.data) if i in child.partitions
        )
    )


#
# PartitionMetadata sub-classes
#


@dataclass(frozen=True)
class RowCountMetadata(PartitionMetadata):
    """Tracks the row count of each partition"""

    def sum(self):
        """Return the total row-count of all partitions"""
        return sum(self.data)


@RowCountMetadata.inherit.register
def _rowcount_elemwise(self, child: Elemwise):
    # All Element-wise operations may inherit
    # row-count metadata "as is"
    return self
