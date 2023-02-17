from __future__ import annotations
from functools import cached_property

from fsspec.utils import stringify_path

import dask
from dask.dataframe.io.parquet.core import (
    get_engine,
    process_statistics,
    set_index_columns,
    ParquetFunctionWrapper,
)
from dask.dataframe.io.parquet.utils import _split_user_options
from dask.utils import natural_sort_key

from dask_match.core import IO


NONE_LABEL = "__null_dask_index__"


class ReadParquet(IO):
    _parameters = [
        "path",
        "columns",
        "filters",
        "categories",
        "index",
        "storage_options",
        "engine",
        "use_nullable_dtypes",
        "calculate_divisions",
        "ignore_metadata_file",
        "metadata_task_size",
        "split_row_groups",
        "blocksize",
        "aggregate_files",
        "parquet_file_extension",
        "filesystem",
        "kwargs",
    ]
    _defaults = {
        "columns": None,
        "filters": None,
        "categories": None,
        "index": None,
        "storage_options": None,
        "engine": "auto",
        "use_nullable_dtypes": False,
        "calculate_divisions": False,
        "ignore_metadata_file": False,
        "metadata_task_size": None,
        "split_row_groups": "infer",
        "blocksize": "default",
        "aggregate_files": None,
        "parquet_file_extension": (".parq", ".parquet", ".pq"),
        "filesystem": None,
        "kwargs": {},
    }

    @cached_property
    def _dataset_info(self):

        if self.use_nullable_dtypes:
            self.use_nullable_dtypes = dask.config.get("dataframe.dtype_backend")

        if isinstance(self.columns, str):
            self.columns = [self.columns]

        if isinstance(self.engine, str):
            self.engine = get_engine(self.engine)

        if hasattr(self.path, "name"):
            self.path = stringify_path(self.path)

        # Process and split user options
        (
            dataset_options,
            read_options,
            open_file_options,
            other_options,
        ) = _split_user_options(**self.kwargs)

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
        if self.index is None:
            # User is allowing auto-detected index
            auto_index_allowed = True
        if self.index and isinstance(self.index, str):
            self.index = [self.index]

        if self.split_row_groups in ("infer", "auto"):
            # Using blocksize to plan partitioning
            if self.blocksize == "default":
                if hasattr(self.engine, "default_blocksize"):
                    self.blocksize = self.engine.default_blocksize()
                else:
                    self.blocksize = "128MiB"
        else:
            # Not using blocksize - Set to `None`
            self.blocksize = None

        dataset_info = self.engine._collect_dataset_info(
            paths,
            fs,
            self.categories,
            self.index,
            self.calculate_divisions,
            self.filters,
            self.split_row_groups,
            self.blocksize,
            self.aggregate_files,
            self.ignore_metadata_file,
            self.metadata_task_size,
            self.parquet_file_extension,
            {
                "read": read_options,
                "dataset": dataset_options,
                **other_options,
            }
        )
        
        # Infer meta, accounting for index and columns arguments.
        meta = self.engine._create_dd_meta(dataset_info, self.use_nullable_dtypes)
        self.index = [self.index] if isinstance(self.index, str) else self.index
        meta, index, columns = set_index_columns(meta, self.index, self.columns, auto_index_allowed)
        if meta.index.name == NONE_LABEL:
            meta.index.name = None
        dataset_info["meta"] = meta
        dataset_info["index"] = index
        dataset_info["columns"] = columns

        return dataset_info

    @property
    def _meta(self):
        return self._dataset_info["meta"]

    @cached_property
    def _plan(self):
        dataset_info = self._dataset_info
        parts, stats, common_kwargs = self.engine._construct_collection_plan(dataset_info)

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
                self.use_nullable_dtypes,
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

    def _layer(self):
        io_func = self._plan["func"]
        parts = self._plan["parts"]
        return {
            (self._name, i): (io_func, part)
            for i, part in enumerate(parts)
        }
