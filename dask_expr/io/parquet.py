from __future__ import annotations

import itertools
import operator
import warnings
from functools import cached_property

from dask.dataframe.io.parquet.core import (
    ParquetFunctionWrapper,
    get_engine,
    process_statistics,
    set_index_columns,
)
from dask.dataframe.io.parquet.utils import _split_user_options
from dask.dataframe.io.utils import _is_local_fs
from dask.utils import natural_sort_key

from dask_expr.expr import EQ, GE, GT, LE, LT, NE, And, Expr, Filter, Or, Projection
from dask_expr.io import BlockwiseIO, PartitionsFiltered

NONE_LABEL = "__null_dask_index__"


def _list_columns(columns):
    # Simple utility to convert columns to list
    if isinstance(columns, (str, int)):
        columns = [columns]
    elif isinstance(columns, tuple):
        columns = list(columns)
    return columns


class ToParquet(Expr):
    _parameters = [
        "frame",
        "path",
        "compression",
        "write_index",
        "append",
        "overwrite",
        "ignore_divisions",
        "partition_on",
        "storage_options",
        "custom_metadata",
        "write_metadata_file",
        "compute",
        "compute_kwargs",
        "schema",
        "name_function",
        "filesystem",
        "kwargs",
    ]
    _defaults = {
        "compression": "snappy",
        "write_index": True,
        "append": False,
        "overwrite": False,
        "ignore_divisions": False,
        "partition_on": None,
        "storage_options": None,
        "custom_metadata": None,
        "write_metadata_file": None,
        "compute": True,
        "compute_kwargs": None,
        "schema": "infer",
        "name_function": None,
        "filesystem": None,
        "kwargs": None,
    }

    @property
    def engine(self):
        return get_engine("pyarrow")

    @property
    def path(self):
        return self._filesystem_info["path"]

    @property
    def fs(self):
        return self._filesystem_info["fs"]

    @cached_property
    def _filesystem_info(self):
        fs, _paths, _, _ = self.engine.extract_filesystem(
            self.operand("path"),
            filesystem=self.operand("filesystem"),
            dataset_options={},
            open_file_options={},
            storage_options=self.operand("storage_options"),
        )
        assert len(_paths) == 1, "only one path"
        path = _paths[0]

        if self.operand("overwrite"):
            if self.operand("append"):
                raise ValueError("Cannot use both `overwrite=True` and `append=True`!")

            if fs.exists(path) and fs.isdir(path):
                # TODO: Need utility to search for specific Expr types
                # # Check for any previous parquet layers reading from a file in the
                # # output directory, since deleting those files now would result in
                # # errors or incorrect results.
                # for layer_name, layer in df.dask.layers.items():
                #     if layer_name.startswith("read-parquet-") and isinstance(
                #         layer, DataFrameIOLayer
                #     ):
                #         path_with_slash = path.rstrip("/") + "/"  # ensure trailing slash
                #         for input in layer.inputs:
                #             # Note that `input` may be either `dict` or `List[dict]`
                #             for piece_dict in input if isinstance(input, list) else [input]:
                #                 if piece_dict["piece"][0].startswith(path_with_slash):
                #                     raise ValueError(
                #                         "Reading and writing to the same parquet file within "
                #                         "the same task graph is not supported."
                #                     )

                # Don't remove the directory if it's the current working directory
                if _is_local_fs(fs):
                    working_dir = fs.expand_path(".")[0]
                    if path.rstrip("/") == working_dir.rstrip("/"):
                        raise ValueError(
                            "Cannot clear the contents of the current working directory!"
                        )

                # It's safe to clear the output directory
                fs.rm(path, recursive=True)

        return {"fs": fs, "path": path}

    def division_info(self):
        # Save divisions and corresponding index name. This is necessary,
        # because we may be resetting the index to write the file
        df = self.frame
        division_info = {"divisions": df.divisions, "name": df.index.name}
        if division_info["name"] is None:
            # As of 0.24.2, pandas will rename an index with name=None
            # when df.reset_index() is called.  The default name is "index",
            # but dask will always change the name to the NONE_LABEL constant
            if NONE_LABEL not in df.columns:
                division_info["name"] = NONE_LABEL
            elif self.operand("write_index"):
                raise ValueError(
                    "Index must have a name if __null_dask_index__ is a column."
                )
            else:
                warnings.warn(
                    "If read back by Dask, column named __null_dask_index__ "
                    "will be set to the index (and renamed to None)."
                )
        return division_info


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

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            # Column projection
            operands = list(self.operands)
            operands[self._parameters.index("columns")] = _list_columns(
                parent.operand("columns")
            )
            if isinstance(parent.operand("columns"), (str, int)):
                operands[self._parameters.index("_series")] = True
            return ReadParquet(*operands)

        if isinstance(parent, Filter) and isinstance(
            parent.predicate, (LE, GE, LT, GT, EQ, NE, And, Or)
        ):
            # Predicate pushdown
            filters = _DNF.extract_pq_filters(self, parent.predicate)
            if filters:
                kwargs = dict(zip(self._parameters, self.operands))
                kwargs["filters"] = filters.combine(kwargs["filters"]).to_list_tuple()
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
# Filters
#


class _DNF:
    """Manage filters in Disjunctive Normal Form (DNF)"""

    class _Or(frozenset):
        """Fozen set of disjunctions"""

        def to_list_tuple(self) -> list:
            # DNF "or" is List[List[Tuple]]
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

    class _And(frozenset):
        """Frozen set of conjunctions"""

        def to_list_tuple(self) -> list:
            # DNF "and" is List[Tuple]
            return tuple(
                val.to_list_tuple() if hasattr(val, "to_list_tuple") else val
                for val in self
            )

    _filters: _And | _Or | None  # Underlying filter expression

    def __init__(self, filters: _And | _Or | list | tuple | None) -> _DNF:
        self._filters = self.normalize(filters)

    def to_list_tuple(self) -> list:
        return self._filters.to_list_tuple()

    def __bool__(self) -> bool:
        return bool(self._filters)

    @classmethod
    def normalize(cls, filters: _And | _Or | list | tuple | None):
        """Convert raw filters to the `_Or(_And)` DNF representation"""
        if not filters:
            result = None
        elif isinstance(filters, list):
            conjunctions = filters if isinstance(filters[0], list) else [filters]
            result = cls._Or([cls._And(conjunction) for conjunction in conjunctions])
        elif isinstance(filters, tuple):
            if isinstance(filters[0], tuple):
                raise TypeError("filters must be List[Tuple] or List[List[Tuple]]")
            result = cls._Or((cls._And((filters,)),))
        elif isinstance(filters, cls._Or):
            result = cls._Or(se for e in filters for se in cls.normalize(e))
        elif isinstance(filters, cls._And):
            total = []
            for c in itertools.product(*[cls.normalize(e) for e in filters]):
                total.append(cls._And(se for e in c for se in e))
            result = cls._Or(total)
        else:
            raise TypeError(f"{type(filters)} not a supported type for _DNF")
        return result

    def combine(self, other: _DNF | _And | _Or | list | tuple | None) -> _DNF:
        """Combine with another _DNF object"""
        if not isinstance(other, _DNF):
            other = _DNF(other)
        assert isinstance(other, _DNF)
        if self._filters is None:
            result = other._filters
        elif other._filters is None:
            result = self._filters
        else:
            result = self._And([self._filters, other._filters])
        return _DNF(result)

    @classmethod
    def extract_pq_filters(cls, pq_expr: ReadParquet, predicate_expr: Expr) -> _DNF:
        _filters = None
        if isinstance(predicate_expr, (LE, GE, LT, GT, EQ, NE)):
            if (
                isinstance(predicate_expr.left, ReadParquet)
                and predicate_expr.left.path == pq_expr.path
                and not isinstance(predicate_expr.right, Expr)
            ):
                op = predicate_expr._operator_repr
                column = predicate_expr.left.columns[0]
                value = predicate_expr.right
                _filters = (column, op, value)
            elif (
                isinstance(predicate_expr.right, ReadParquet)
                and predicate_expr.right.path == pq_expr.path
                and not isinstance(predicate_expr.left, Expr)
            ):
                # Simple dict to make sure field comes first in filter
                flip = {LE: GE, LT: GT, GE: LE, GT: LT}
                op = predicate_expr
                op = flip.get(op, op)._operator_repr
                column = predicate_expr.right.columns[0]
                value = predicate_expr.left
                _filters = (column, op, value)

        elif isinstance(predicate_expr, (And, Or)):
            left = cls.extract_pq_filters(pq_expr, predicate_expr.left)._filters
            right = cls.extract_pq_filters(pq_expr, predicate_expr.right)._filters
            if left and right:
                if isinstance(predicate_expr, And):
                    _filters = cls._And([left, right])
                else:
                    _filters = cls._Or([left, right])

        return _DNF(_filters)
