import operator
import numpy as np

from dask.dataframe.core import _concat
from dask.dataframe.shuffle import partitioning_index, shuffle_group

from dask_match.expr import Assign, Expr, Blockwise


class Shuffle(Expr):
    """Abstract shuffle class

    Parameters
    ----------
    frame: Expr
        The DataFrame-like expression to shuffle.
    partitioning_index: str, list
        Column and/or index names to hash and partition by.
    npartitions: int
        Number of output partitions.
    ignore_index: bool
        Whether to ignore the index during this shuffle operation.
    backend: str or Callable
        Label or callback funcition to convert a shuffle operation
        to its necessary components.
    options: dict
        Algorithm-specific options.
    """

    _parameters = [
        "frame",
        "partitioning_index",
        "npartitions_out",
        "ignore_index",
        "backend",
        "options",
    ]

    def __str__(self):
        return f"Shuffle({self._name[-7:]})"

    def simplify(self, lower: bool = False):
        if lower is False:
            return None
        # Use `backend` to decide how to compose a
        # shuffle operation from concerete expressions
        backend = self.backend or "simple"
        if isinstance(backend, ShuffleBackend):
            lower = backend.from_abstract_shuffle
        elif backend == "simple":
            # Only support "SimpleShuffle" for now
            lower = SimpleShuffle.from_abstract_shuffle
        else:
            raise ValueError(f"{backend} not supported")
        return lower(self)

    def _layer(self):
        raise NotImplementedError(
            f"{self} is abstract! Please call `simplify`"
            f"before generating a task graph."
        )

    @property
    def _meta(self):
        return self.frame._meta

    def _divisions(self):
        return (None,) * (self.npartitions_out + 1)


#
# ShuffleBackend
#


class ShuffleBackend(Shuffle):
    """Base shuffle-backend class"""

    _parameters = [
        "frame",
        "partitioning_index",
        "npartitions_out",
        "ignore_index",
        "options",
    ]

    @classmethod
    def from_abstract_shuffle(cls, expr: Shuffle) -> Expr:
        """Create an Expr tree that uses this ShuffleBackend class"""
        raise NotImplementedError()

    def simplify(self, lower: bool = False):
        return None


#
# SimpleShuffle
#


class SimpleShuffle(ShuffleBackend):
    """Simple task-based shuffle implementation"""

    @classmethod
    def from_abstract_shuffle(cls, expr: Shuffle) -> Expr:
        frame = expr.frame
        partitioning_index = expr.partitioning_index
        npartitions_out = expr.npartitions_out
        ignore_index = expr.ignore_index
        options = expr.options

        # Normalize partitioning_index
        if isinstance(partitioning_index, str):
            partitioning_index = [partitioning_index]
        if not isinstance(partitioning_index, list):
            raise ValueError(
                f"{type(partitioning_index)} not a supported type for partitioning_index"
            )
        partitioning_index = _select_columns_or_index(frame, partitioning_index)

        # Assign partitioning-index as a new "_partitions" column
        index_added = Assign(
            frame,
            "_partitions",
            PartitioningIndex(frame, partitioning_index, npartitions_out),
        )

        # Apply shuffle
        shuffled = cls(
            index_added,
            "_partitions",
            npartitions_out,
            ignore_index,
            options,
        )

        # Drop "_partitions" column and return
        return shuffled[[c for c in shuffled.columns if c != "_partitions"]]

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
                        self.partitioning_index,
                        0,
                        self.npartitions,
                        self.npartitions,
                        self.ignore_index,
                        self.npartitions,
                    )

        return dsk


#
# Helper logic
#


def make_partitioning_index(df, index, npartitions: int):
    """Construct a hash-based partitioning index"""
    if isinstance(index, (str, list, tuple)):
        # Assume column selection from df
        index = [index] if isinstance(index, str) else list(index)
        return partitioning_index(df[index], npartitions)
    return partitioning_index(index, npartitions)


def _select_columns_or_index(expr, columns_or_index):
    """
    Make a column selection that may include the index

    Parameters
    ----------
    columns_or_index
        Column or index name, or a list of these
    """

    # Ensure columns_or_index is a list
    columns_or_index = (
        columns_or_index if isinstance(columns_or_index, list) else [columns_or_index]
    )

    column_names = [n for n in columns_or_index if _is_column_label_reference(expr, n)]

    selected_expr = expr[column_names]
    if _contains_index_name(expr, columns_or_index):
        # Index name was included
        selected_expr = Assign(selected_expr, "_index", expr.index)

    return selected_expr


def _is_column_label_reference(expr, key):
    """
    Test whether a key is a column label reference

    To be considered a column label reference, `key` must match the name of at
    least one column.
    """
    return (
        not isinstance(key, Expr)
        and (np.isscalar(key) or isinstance(key, tuple))
        and key in expr.columns
    )


def _contains_index_name(expr, columns_or_index):
    """
    Test whether the input contains a reference to the index of the Expr
    """
    if isinstance(columns_or_index, list):
        return any(_is_index_level_reference(expr, n) for n in columns_or_index)
    else:
        return _is_index_level_reference(expr, columns_or_index)


def _is_index_level_reference(expr, key):
    """
    Test whether a key is an index level reference

    To be considered an index level reference, `key` must match the index name
    and must NOT match the name of any column.
    """
    return (
        expr.index.name is not None
        and not isinstance(key, Expr)
        and (np.isscalar(key) or isinstance(key, tuple))
        and key == expr.index.name
        and key not in getattr(expr, "columns", ())
    )


class PartitioningIndex(Blockwise):
    """Create a partitioning index

    This class is used to construct a hash-based
    partitioning index for shuffling.

    Parameters
    ----------
    frame: Expr
        Frame-like expression being partitioned.
    index: Expr or list
        Index-like expression or list of columns to construct
        the partitioning-index from.
    npartitions_out: int
        Number of partitions after repartitioning is finished.
    """

    _parameters = ["frame", "index", "npartitions_out"]
    operation = make_partitioning_index

    @property
    def _meta(self):
        index = self.operand("index")
        if isinstance(index, Expr):
            index = index._meta
        return make_partitioning_index(self.frame._meta, index, self.npartitions_out)

    def _task(self, index: int):
        partition_index = index
        index = self.operand("index")
        if isinstance(index, Expr):
            index = (index._name, partition_index)
        return (
            make_partitioning_index,
            (self.frame._name, partition_index),
            index,
            self.npartitions_out,
        )
