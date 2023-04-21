import functools
from operator import getitem

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from dask.dataframe import methods
from dask.dataframe.core import split_evenly

from dask_match.expr import Expr


class Repartition(Expr):
    _parameters = ["frame", "n", "new_divisions"]

    @property
    def _meta(self):
        return self.frame._meta

    def _divisions(self):
        if self.n is not None:
            return (None,) * (self.n + 1)
        return self.new_divisions

    def simplify(self):
        if self.n is not None:
            if self.n < self.frame.npartitions:
                return ReducePartitionCount(self.frame, self.n)
            else:
                divisions = pd.Series(self.frame.divisions).drop_duplicates()
                if self.frame.known_divisions and (
                    is_datetime64_any_dtype(divisions.dtype)
                    or is_numeric_dtype(divisions.dtype)
                ):
                    # Need to repartition by divisions
                    raise NotImplementedError()  # TODO: repartition by divisions
                return IncreasePartitionCount(self.frame, self.n)
        raise NotImplementedError()  # TODO: repartition by divisions


class ReducePartitionCount(Repartition):
    _parameters = ["frame", "n"]

    def _divisions(self):
        return tuple(self.frame.divisions[i] for i in self._partitions_boundaries)

    @functools.cached_property
    def _partitions_boundaries(self):
        npartitions = self.n
        npartitions_input = self.frame.npartitions
        assert npartitions_input > self.n

        npartitions_ratio = npartitions_input / npartitions
        new_partitions_boundaries = [
            int(new_partition_index * npartitions_ratio)
            for new_partition_index in range(npartitions + 1)
        ]

        if not isinstance(new_partitions_boundaries, list):
            new_partitions_boundaries = list(new_partitions_boundaries)
        if new_partitions_boundaries[0] > 0:
            new_partitions_boundaries.insert(0, 0)
        if new_partitions_boundaries[-1] < self.frame.npartitions:
            new_partitions_boundaries.append(self.frame.npartitions)
        return new_partitions_boundaries

    def _layer(self):
        new_partitions_boundaries = self._partitions_boundaries
        return {
            (self._name, i): (
                methods.concat,
                [(self.frame._name, j) for j in range(start, end)],
            )
            for i, (start, end) in enumerate(
                zip(new_partitions_boundaries, new_partitions_boundaries[1:])
            )
        }


class IncreasePartitionCount(Repartition):
    _parameters = ["frame", "n"]

    def _divisions(self):
        return (None,) * (1 + sum(self._nsplits))

    @functools.cached_property
    def _nsplits(self):
        df = self.frame
        div, mod = divmod(self.n, df.npartitions)
        nsplits = [div] * df.npartitions
        nsplits[-1] += mod
        if len(nsplits) != df.npartitions:
            raise ValueError(f"nsplits should have len={df.npartitions}")
        return nsplits

    def _layer(self):
        dsk = {}
        nsplits = self._nsplits
        df = self.frame
        new_name = self._name
        split_name = f"split-{new_name}"
        j = 0
        for i, k in enumerate(nsplits):
            if k == 1:
                dsk[new_name, j] = (df._name, i)
                j += 1
            else:
                dsk[split_name, i] = (split_evenly, (df._name, i), k)
                for jj in range(k):
                    dsk[new_name, j] = (getitem, (split_name, i), jj)
                    j += 1
        return dsk
