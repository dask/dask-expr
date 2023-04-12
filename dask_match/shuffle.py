import operator

from dask.dataframe.core import _concat
from dask.dataframe.shuffle import partitioning_index, shuffle_group

from dask_match.expr import Expr, Blockwise


def make_partitioning_index(df, index: str | list, npartitions: int):
    """Construct a hash-based partitioning index"""

    # TODO: Support cases other than index is column list
    index = [index] if isinstance(index, str) else list(index)
    return partitioning_index(df[index], npartitions)


class PartitioningIndex(Blockwise):
    """Create a partitioning index

    This class is used by shuffle routines to assign a
    hash-based partitioning-index as a new column.

    Parameters
    ----------
    frame: Expr
        Frame-like expression being partitioned.
    column_name: str
        Column name to use for the partitioning-index.
    npartitions: int
        Number of partitions after repartitioning is finished.
    """
    _parameters = ["frame", "column_name", "npartitions"]
    operation = make_partitioning_index

    @property
    def _meta(self):
        return make_partitioning_index(self.frame._meta, self.column_name, self.npartitions)

    def _blockwise_layer(self):
        return {
            self._name: (
                make_partitioning_index,
                self.frame._name,
                self.column_name,
                self.npartitions,
            )
        }


class BaseShuffle(Expr):
    """Base shuffle class
    
    TODO: Should allow ``keys`` to be a list of column names.
    Dask-cudf performance is typically better when we avoid
    assigning a "_partitions" column up front, and instead
    perform the hash+modulus within ``shuffle_group``.

    Parameters
    ----------
    frame: Expr
        The DataFrame-like expression to shuffle.
    keys: str
        Name of the column to shuffle by. We assume that the values
        of ``frame[keys]`` already correspond to the final partition
        indices for every row of ``frame``.
    npartitions: int
        Number of output partitions.
    ignore_index: bool
        Whether to ignore the index during this shuffle operation.
    options: dict
        Algorithm-specific options.
    """
    _parameters = ["frame", "keys", "npartitions", "ignore_index", "options"]

    def _layer(self):
        raise NotImplementedError

    @property
    def _meta(self):
        return self.frame._meta

    def _divisions(self):
        return (None,) * (self.npartitions + 1)
    

class SimpleShuffle(BaseShuffle):
    """Simple task-based shuffle implementation"""

    def _layer(self):
        """Construct graph for a simple shuffle operation."""
        shuffle_group_name = "group-" + self._name
        split_name = "split-" + self._name

        dsk = {}
        for part_out in range(self.npartitions):
            _concat_list = [
                (split_name, part_out, part_in)
                for part_in in range(self.frame.npartitions)
            ]
            dsk[(self._name, part_out)] = (
                _concat,
                _concat_list,
                self.ignore_index,
            )
            for _, _part_out, _part_in in _concat_list:
                dsk[(split_name, _part_out, _part_in)] = (
                    operator.getitem,
                    (shuffle_group_name, _part_in),
                    _part_out,
                )
                if (shuffle_group_name, _part_in) not in dsk:
                    dsk[(shuffle_group_name, _part_in)] = (
                        shuffle_group,
                        (self.frame._name, _part_in),
                        self.keys,
                        0,
                        self.npartitions,
                        self.npartitions,
                        self.ignore_index,
                        self.npartitions,
                    )

        return dsk