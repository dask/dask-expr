from __future__ import annotations

import enum
import math
from enum import IntEnum
from functools import cached_property

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dask._task_spec import Task
from dask.typing import Key
from dask.utils import funcname, parse_bytes

from dask_expr._expr import Index, Projection, determine_column_projection
from dask_expr._util import _convert_to_list, _tokenize_deterministic
from dask_expr.io import BlockwiseIO, PartitionsFiltered
from dask_expr.io.parquet import _maybe_adjust_cpu_count


class PartitionFlavor(IntEnum):
    """Flavor of file:partition mapping."""

    SINGLE_FILE = enum.auto()  # 1:1 mapping between files and partitions
    SPLIT_FILES = enum.auto()  # Split each file into >1 partition
    FUSED_FILES = enum.auto()  # Fuse multiple files into each partition


class PartitionPlan:
    """Partition-mappiing plan."""

    __slots__ = ("factor", "flavor", "count")
    factor: int
    flavor: PartitionFlavor
    count: int

    def __init__(self, factor: int, flavor: PartitionFlavor, file_count: int) -> None:
        if flavor == PartitionFlavor.SINGLE_FILE and factor != 1:
            raise ValueError(f"Expected factor == 1 for {flavor}, got: {factor}")
        self.factor = factor
        self.flavor = flavor
        if flavor == PartitionFlavor.SINGLE_FILE:
            self.count = file_count
        elif flavor == PartitionFlavor.SPLIT_FILES:
            self.count = file_count * factor
        elif flavor == PartitionFlavor.FUSED_FILES:
            self.count = math.ceil(file_count / factor)
        else:
            raise ValueError(f"{flavor} not a supported PartitionFlavor")


def _pa_filters(filters):
    # Simple utility to covert filters to
    # a pyarrow-compatible expression
    if filters is None:
        return None
    else:
        return pq.filters_to_expression(filters)


class FromArrowDataset(PartitionsFiltered, BlockwiseIO):
    _parameters = [
        "dataset",
        "columns",
        "filters",
        "blocksize",
        "path_column",
        "fragment_to_table_options",
        "table_to_pandas_options",
        "custom_backend_options",
        "_partitions",
        "_series",
    ]
    _defaults = {
        "columns": None,
        "filters": None,
        "blocksize": None,
        "path_column": None,
        "fragment_to_table_options": None,
        "table_to_pandas_options": None,
        "custom_backend_options": None,
        "_partitions": None,
        "_series": False,
    }

    _absorb_projections = True
    _filter_passthrough = False
    _scan_options = None

    @property
    def columns(self):
        columns_operand = self.operand("columns")
        if columns_operand is None:
            return list(self._meta.columns)
        else:
            return _convert_to_list(columns_operand)

    @cached_property
    def pa_filters(self):
        return _pa_filters(self.filters)

    @cached_property
    def _name(self):
        return "from-dataset-" + _tokenize_deterministic(
            funcname(type(self)), *self.operands[:-1]
        )

    @cached_property
    def _meta(self):
        schema = self.dataset.schema
        if self.path_column is not None:
            if self.path_column in schema.names:
                raise ValueError(f"{self.path_column} column already exists in schema.")
            schema = schema.append(pa.field(self.path_column, pa.string()))
        meta = schema.empty_table().to_pandas()
        columns = _convert_to_list(self.operand("columns"))
        if self._series:
            assert len(columns) > 0
            return meta[columns[0]]
        elif columns is not None:
            return meta[columns]
        return meta

    @cached_property
    def _mean_file_size(self):
        return np.mean(
            [
                frag.filesystem.get_file_info(frag.path).size
                for frag in self.fragments[:3]
            ]
        )

    @cached_property
    def _plan(self) -> PartitionPlan:
        num_files = len(self.fragments)
        plan = PartitionPlan(
            factor=1,
            flavor=PartitionFlavor.SINGLE_FILE,
            file_count=num_files,
        )  # Default plan
        blocksize = self.operand("blocksize")
        if blocksize is not None:
            blocksize = parse_bytes(blocksize)
            # TODO: Use metadata for Parquet
            file_size = self._mean_file_size
            if file_size > blocksize:
                # Split large files
                plan = PartitionPlan(
                    factor=math.ceil(file_size / blocksize),
                    flavor=PartitionFlavor.SPLIT_FILES,
                    file_count=num_files,
                )
            else:
                # Aggregate small files
                plan = PartitionPlan(
                    factor=max(int(blocksize / file_size), 1),
                    flavor=PartitionFlavor.FUSED_FILES,
                    file_count=num_files,
                )
        return plan

    def _divisions(self):
        return (None,) * (self._plan.count + 1)

    @staticmethod
    def _table_to_pandas(table):
        return table.to_pandas()

    @cached_property
    def fragments(self):
        if self.pa_filters is not None:
            return np.array(list(self.dataset.get_fragments(filter=self.pa_filters)))
        return np.array(list(self.dataset.get_fragments()))

    @classmethod
    def read_fragments(
        cls,
        fragments,
        columns,
        filters,
        schema,
        path_column,
        fragment_to_table_options,
        table_to_pandas_options,
        custom_backend_options,
        split_range,
    ):
        """Read list of fragments into DataFrame partitions."""
        fragment_to_table_options = fragment_to_table_options or {}
        table_to_pandas_options = table_to_pandas_options or {}
        if custom_backend_options:
            raise ValueError(f"Unsupported options: {custom_backend_options}")
        return cls._table_to_pandas(
            pa.concat_tables(
                [
                    cls._fragment_to_table(
                        fragment,
                        filters=filters,
                        columns=columns,
                        schema=schema,
                        split_range=split_range,
                        path_column=path_column,
                        **fragment_to_table_options,
                    )
                    for fragment in fragments
                ],
                promote_options="permissive",
            ),
            **table_to_pandas_options,
        )

    def _filtered_task(self, name: Key, index: int) -> Task:
        columns = self.columns.copy()
        schema = self.dataset.schema.remove_metadata()
        flavor = self._plan.flavor
        if flavor == PartitionFlavor.SPLIT_FILES:
            splits = self._plan.factor
            stride = 1
            frag_index = int(index / splits)
            split_range = (index % splits, splits)
        else:
            stride = self._plan.factor
            frag_index = index * stride
            split_range = None
        return Task(
            name,
            self.read_fragments,
            self.fragments[frag_index : frag_index + stride],
            columns=columns,
            filters=self.filters,
            schema=schema,
            path_column=self.path_column,
            fragment_to_table_options=self.fragment_to_table_options,
            table_to_pandas_options=self.table_to_pandas_options,
            custom_backend_options=self.custom_backend_options,
            split_range=split_range,
        )

    @classmethod
    def _fragment_to_table(
        cls,
        fragment,
        filters,
        columns,
        schema,
        split_range,
        path_column,
        **fragment_to_table_options,
    ):
        _maybe_adjust_cpu_count()
        options = {
            "columns": (
                columns
                if path_column is None
                else [c for c in columns if c != path_column]
            ),
            "filter": _pa_filters(filters),
            "batch_size": 10_000_000,
            "fragment_scan_options": cls._scan_options,
            "use_threads": True,
        }
        options.update(fragment_to_table_options)
        if split_range:
            total_rows = fragment.count_rows(filter=filters)
            n_rows = int(total_rows / split_range[1])
            skip_rows = n_rows * split_range[0]
            if split_range[0] == (split_range[1] - 1):
                end = total_rows
            else:
                end = skip_rows + n_rows
            table = fragment.take(
                range(skip_rows, end),
                **options,
            )
        else:
            table = fragment.to_table(
                schema=schema,
                **options,
            )
        if path_column is None:
            return table
        else:
            return table.append_column(
                path_column, pa.array([fragment.path] * len(table), pa.string())
            )

    @property
    def _fusion_compression_factor(self):
        # TODO: Deal with column-projection adjustments
        return 1

    def _simplify_up(self, parent, dependents):
        if isinstance(parent, Index):
            # Column projection
            columns = determine_column_projection(self, parent, dependents)
            columns = [col for col in self.columns if col in columns]
            if set(columns) == set(self.columns):
                return
            return Index(
                self.substitute_parameters({"columns": columns, "_series": False})
            )

        if isinstance(parent, Projection):
            return super()._simplify_up(parent, dependents)

    def _simplify_down(self):
        file_format = self.dataset.format.default_extname
        if file_format == "parquet":
            return FromArrowDatasetParquet(*self.operands)


class FromArrowDatasetParquet(FromArrowDataset):
    _scan_options = pa.dataset.ParquetFragmentScanOptions(
        pre_buffer=True,
        cache_options=pa.CacheOptions(
            hole_size_limit=parse_bytes("4 MiB"),
            range_size_limit=parse_bytes("32.00 MiB"),
        ),
    )
