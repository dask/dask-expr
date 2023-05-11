from __future__ import annotations

import itertools
import operator
from functools import cached_property

from dask.dataframe.io.parquet.core import (
    ParquetFunctionWrapper,
    get_engine,
    process_statistics,
    set_index_columns,
)
from dask.dataframe.io.parquet.utils import _split_user_options
from dask.utils import natural_sort_key

from dask_expr.expr import AND, EQ, GE, GT, LE, LT, NE, OR, Expr, Filter, Projection
from dask_expr.io import BlockwiseIO, PartitionsFiltered

NONE_LABEL = "__null_dask_index__"


def _list_columns(columns):
    # Simple utility to convert columns to list
    if isinstance(columns, (str, int)):
        columns = [columns]
    elif isinstance(columns, tuple):
        columns = list(columns)
    return columns


class ReadParquet(PartitionsFiltered, BlockwiseIO):
    """Read a parquet dataset"""

    _parameters = [
        "path",
        "columns",
        "filters",
        "categories",
        "index",
        "storage_options",
        "calculate_divisions",
        "ignore_metadata_file",
        "metadata_task_size",
        "split_row_groups",
        "blocksize",
        "aggregate_files",
        "parquet_file_extension",
        "filesystem",
        "kwargs",
        "_partitions",
        "_series",
    ]
    _defaults = {
        "columns": None,
        "filters": None,
        "categories": None,
        "index": None,
        "storage_options": None,
        "calculate_divisions": False,
        "ignore_metadata_file": False,
        "metadata_task_size": None,
        "split_row_groups": "infer",
        "blocksize": "default",
        "aggregate_files": None,
        "parquet_file_extension": (".parq", ".parquet", ".pq"),
        "filesystem": "fsspec",
        "kwargs": None,
        "_partitions": None,
        "_series": False,
    }

    @property
    def engine(self):
        return get_engine("pyarrow")

    @property
    def columns(self):
        columns_operand = self.operand("columns")
        if columns_operand is None:
            return self._meta.columns
        else:
            import pandas as pd

            return pd.Index(_list_columns(columns_operand))

    def _extract_filter(self, predicate) -> list | None:
        # Extract the List[List[Tuple]] filter expression
        # from a predicate-based Expr object
        if isinstance(predicate, (LE, GE, LT, GT, EQ, NE)):
            if (
                isinstance(predicate.left, ReadParquet)
                and predicate.left.path == self.path
                and not isinstance(predicate.right, Expr)
            ):
                op = predicate._operator_repr
                column = predicate.left.columns[0]
                value = predicate.right
                return [[(column, op, value)]]
            elif (
                isinstance(predicate.right, ReadParquet)
                and predicate.right.path == self.path
                and not isinstance(predicate.left, Expr)
            ):
                # Simple dict to make sure field comes first in filter
                flip = {LE: GE, LT: GT, GE: LE, GT: LT}
                op = predicate
                op = flip.get(op, op)._operator_repr
                column = predicate.right.columns[0]
                value = predicate.left
                return [[(column, op, value)]]
        elif isinstance(predicate, (AND, OR)):
            left = self._extract_filter(predicate.left)
            right = self._extract_filter(predicate.right)
            if left and right:
                left = to_dnf(left)
                right = to_dnf(right)
                if isinstance(predicate, AND):
                    filter = And([left, right])
                else:
                    filter = Or([left, right])
                return to_dnf(filter).to_list_tuple()

        return None

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            operands = list(self.operands)
            operands[self._parameters.index("columns")] = _list_columns(
                parent.operand("columns")
            )
            if isinstance(parent.operand("columns"), (str, int)):
                operands[self._parameters.index("_series")] = True
            return ReadParquet(*operands)

        if isinstance(parent, Filter) and isinstance(
            parent.predicate, (LE, GE, LT, GT, EQ, NE, AND, OR)
        ):
            conjunction = self._extract_filter(parent.predicate)
            if conjunction:
                kwargs = dict(zip(self._parameters, self.operands))
                filters = _add_filter(kwargs["filters"], conjunction)
                kwargs["filters"] = filters
                return ReadParquet(**kwargs)

    @cached_property
    def _dataset_info(self):
        # Process and split user options
        (
            dataset_options,
            read_options,
            open_file_options,
            other_options,
        ) = _split_user_options(**(self.kwargs or {}))

        # Extract global filesystem and paths
        fs, paths, dataset_options, open_file_options = self.engine.extract_filesystem(
            self.path,
            self.filesystem,
            dataset_options,
            open_file_options,
            self.storage_options,
        )
        read_options["open_file_options"] = open_file_options
        paths = sorted(paths, key=natural_sort_key)  # numeric rather than glob ordering

        auto_index_allowed = False
        index_operand = self.operand("index")
        if index_operand is None:
            # User is allowing auto-detected index
            auto_index_allowed = True
        if index_operand and isinstance(index_operand, str):
            index = [index_operand]
        else:
            index = index_operand

        blocksize = self.blocksize
        if self.split_row_groups in ("infer", "adaptive"):
            # Using blocksize to plan partitioning
            if self.blocksize == "default":
                if hasattr(self.engine, "default_blocksize"):
                    blocksize = self.engine.default_blocksize()
                else:
                    blocksize = "128MiB"
        else:
            # Not using blocksize - Set to `None`
            blocksize = None

        dataset_info = self.engine._collect_dataset_info(
            paths,
            fs,
            self.categories,
            index,
            self.calculate_divisions,
            self.filters,
            self.split_row_groups,
            blocksize,
            self.aggregate_files,
            self.ignore_metadata_file,
            self.metadata_task_size,
            self.parquet_file_extension,
            {
                "read": read_options,
                "dataset": dataset_options,
                **other_options,
            },
        )

        # Infer meta, accounting for index and columns arguments.
        meta = self.engine._create_dd_meta(dataset_info)
        index = [index] if isinstance(index, str) else index
        meta, index, columns = set_index_columns(
            meta, index, self.operand("columns"), auto_index_allowed
        )
        if meta.index.name == NONE_LABEL:
            meta.index.name = None
        dataset_info["meta"] = meta
        dataset_info["index"] = index
        dataset_info["columns"] = columns

        return dataset_info

    @property
    def _meta(self):
        meta = self._dataset_info["meta"]
        if self._series:
            column = _list_columns(self.operand("columns"))[0]
            return meta[column]
        return meta

    @cached_property
    def _plan(self):
        dataset_info = self._dataset_info
        parts, stats, common_kwargs = self.engine._construct_collection_plan(
            dataset_info
        )

        # Parse dataset statistics from metadata (if available)
        parts, divisions, _ = process_statistics(
            parts,
            stats,
            dataset_info["filters"],
            dataset_info["index"],
            (
                dataset_info["blocksize"]
                if dataset_info["split_row_groups"] is True
                else None
            ),
            dataset_info["split_row_groups"],
            dataset_info["fs"],
            dataset_info["aggregation_depth"],
        )

        meta = dataset_info["meta"]
        if len(divisions) < 2:
            # empty dataframe - just use meta
            divisions = (None, None)
            io_func = lambda x: x
            parts = [meta]
        else:
            # Use IO function wrapper
            io_func = ParquetFunctionWrapper(
                self.engine,
                dataset_info["fs"],
                meta,
                dataset_info["columns"],
                dataset_info["index"],
                dataset_info["kwargs"]["dtype_backend"],
                {},  # All kwargs should now be in `common_kwargs`
                common_kwargs,
            )

        return {
            "func": io_func,
            "parts": parts,
            "divisions": divisions,
        }

    def _divisions(self):
        return self._plan["divisions"]

    def _filtered_task(self, index: int):
        tsk = (self._plan["func"], self._plan["parts"][index])
        if self._series:
            return (operator.getitem, tsk, self.columns[0])
        return tsk


#
# Filtering/Predicate utilities
#


class Or(frozenset):
    """Helper class for filter disjunctions"""

    def to_list_tuple(self):
        # NDF "or" is List[List[Tuple]]
        def _maybe_list(val):
            if isinstance(val, tuple) and val and isinstance(val[0], (tuple, list)):
                return list(val)
            return [val]

        return [
            _maybe_list(val.to_list_tuple())
            if hasattr(val, "to_list_tuple")
            else _maybe_list(val)
            for val in self
        ]


class And(frozenset):
    """Helper class for filter conjunctions"""

    def to_list_tuple(self):
        # NDF "and" is List[Tuple]
        return tuple(
            val.to_list_tuple() if hasattr(val, "to_list_tuple") else val
            for val in self
        )


def to_dnf(filter: Or | And | list | tuple | None) -> Or | None:
    """Normalize a filter expression to disjunctive normal form (DNF)

    This function will always convert the provided expression
    into `Or((And(...), ...))`. Call `to_list_tuple` on the
    result to translate the filters into `List[List[Tuple]]`.
    """

    # Credit: https://stackoverflow.com/a/58372345
    if not filter:
        result = None
    elif isinstance(filter, Or):
        result = Or(se for e in filter for se in to_dnf(e))
    elif isinstance(filter, And):
        total = []
        for c in itertools.product(*[to_dnf(e) for e in filter]):
            total.append(And(se for e in c for se in e))
        result = Or(total)
    elif isinstance(filter, list):
        disjunction = []
        stack = filter.copy()
        while stack:
            conjunction, *stack = stack if isinstance(stack[0], list) else [stack]
            disjunction.append(And(conjunction))
        result = Or(disjunction)
    elif isinstance(filter, tuple):
        if isinstance(filter[0], tuple):
            raise TypeError("filters must be List[Tuple] or List[List[Tuple]]")
        result = Or((And((filter,)),))
    else:
        raise TypeError(f"{type(filter)} not a supported type for to_dnf")
    return result


def _add_filter(old_filters: list, new_filter: list | tuple) -> list:
    # Add a new filter to an existing filters expression.
    disjunctions = to_dnf(old_filters)
    new = to_dnf(new_filter)
    if disjunctions is None:
        result = new
    else:
        result = And([disjunctions, new])
    return to_dnf(result).to_list_tuple()
