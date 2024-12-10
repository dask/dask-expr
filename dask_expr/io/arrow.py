from __future__ import annotations

import math
from functools import cached_property

import numpy as np
import pyarrow as pa
from dask._task_spec import Task
from dask.typing import Key
from dask.utils import funcname, parse_bytes

from dask_expr._util import _convert_to_list, _tokenize_deterministic
from dask_expr.io import BlockwiseIO, PartitionsFiltered
from dask_expr.io.parquet import _maybe_adjust_cpu_count


class FromArrowDataset(PartitionsFiltered, BlockwiseIO):
    _parameters = [
        "dataset",
        "columns",
        "filters",
        "blocksize",
        "_partitions",
        "_series",
    ]
    _defaults = {
        "columns": None,
        "filters": None,
        "blocksize": None,
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
    def _name(self):
        return (
            "from-dataset"
            + "-"
            + _tokenize_deterministic(funcname(type(self)), *self.operands[:-1])
        )

    @cached_property
    def _meta(self):
        meta = self.dataset.schema.empty_table().to_pandas()
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
    def _plan(self):
        # TODO: Use metadata for Parquet
        num_files = len(self.fragments)
        splits, stride = 1, 1
        blocksize = self.operand("blocksize")
        if blocksize:
            file_size = self._mean_file_size
            if file_size > blocksize:
                # Split large files
                splits = math.ceil(file_size / blocksize)
            else:
                # Aggregate small files
                stride = max(int(blocksize / file_size), 1)
        if splits > 1:
            count = num_files * splits
        else:
            count = math.ceil(num_files / stride)
        return {
            "count": count,
            "splits": splits,
            "stride": stride,
        }

    def _divisions(self):
        return (None,) * (self._plan["count"] + 1)

    @staticmethod
    def _table_to_pandas(table):
        return table.to_pandas()

    @cached_property
    def fragments(self):
        if self.filters is not None:
            return np.array(list(self.dataset.get_fragments(filter=self.filters)))
        return np.array(list(self.dataset.get_fragments()))

    @classmethod
    def _read_fragments(
        cls,
        fragments,
        columns,
        filters,
        schema,
        fragment_to_table_options,
        table_to_pandas_options,
        split_range,
    ):
        fragment_to_table_options = fragment_to_table_options or {}
        table_to_pandas_options = table_to_pandas_options or {}
        return cls._table_to_pandas(
            pa.concat_tables(
                [
                    cls._fragment_to_table(
                        fragment,
                        filters=filters,
                        columns=columns,
                        schema=schema,
                        split_range=split_range,
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
        splits, stride = self._plan["splits"], self._plan["stride"]
        if splits > 1:
            frag_index = int(index / splits)
            split_range = (index % splits, splits)
        else:
            frag_index = index * stride
            split_range = None
        return Task(
            name,
            self._read_fragments,
            self.fragments[frag_index : frag_index + stride],
            columns=columns,
            filters=self.filters,
            schema=schema,
            fragment_to_table_options=None,
            table_to_pandas_options=None,
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
        **fragment_to_table_options,
    ):
        _maybe_adjust_cpu_count()
        options = {
            "columns": columns,
            "filter": filters,
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
            return fragment.take(
                range(skip_rows, end),
                **options,
            )
        else:
            return fragment.to_table(
                schema=schema,
                **options,
            )


class FromArrowDatasetParquet(FromArrowDataset):
    _scan_options = pa.dataset.ParquetFragmentScanOptions(
        pre_buffer=True,
        cache_options=pa.CacheOptions(
            hole_size_limit=parse_bytes("4 MiB"),
            range_size_limit=parse_bytes("32.00 MiB"),
        ),
    )
