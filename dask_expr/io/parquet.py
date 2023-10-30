from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import itertools
import math
import warnings
from collections import defaultdict
from functools import cached_property

import dask
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import tlz as toolz
from dask.base import normalize_token
from dask.dataframe.io.parquet.core import ToParquetFunctionWrapper, get_engine
from dask.dataframe.io.utils import _is_local_fs
from dask.delayed import delayed
from dask.utils import apply, typename
from fsspec.utils import stringify_path

from dask_expr._expr import Blockwise, Expr, Index, PartitionsFiltered, Projection
from dask_expr._util import _convert_to_list, _tokenize_deterministic
from dask_expr.io import BlockwiseIO

NONE_LABEL = "__null_dask_index__"

_cached_dataset_info = {}
_CACHED_DATASET_SIZE = 10
_CACHED_PLAN_SIZE = 10
_cached_plan = {}


def _control_cached_dataset_info(key):
    if (
        len(_cached_dataset_info) > _CACHED_DATASET_SIZE
        and key not in _cached_dataset_info
    ):
        key_to_pop = list(_cached_dataset_info.keys())[0]
        _cached_dataset_info.pop(key_to_pop)


def _control_cached_plan(key):
    if len(_cached_plan) > _CACHED_PLAN_SIZE and key not in _cached_plan:
        key_to_pop = list(_cached_plan.keys())[0]
        _cached_plan.pop(key_to_pop)


@normalize_token.register(pa_ds.Dataset)
def normalize_pa_ds(ds):
    return (ds.files, ds.schema)


@normalize_token.register(pa_ds.FileFormat)
def normalize_pa_file_format(file_format):
    return str(file_format)


@normalize_token.register(pa.Schema)
def normalize_pa_schema(schema):
    return schema.to_string()


class ToParquet(Expr):
    _parameters = [
        "frame",
        "path",
        "fs",
        "fmd",
        "engine",
        "offset",
        "partition_on",
        "write_metadata_file",
        "name_function",
        "write_kwargs",
    ]

    @property
    def _meta(self):
        return None

    def _divisions(self):
        return (None, None)

    def _lower(self):
        return ToParquetBarrier(
            ToParquetData(
                *self.operands,
            ),
            *self.operands[1:],
        )


class ToParquetData(Blockwise):
    _parameters = ToParquet._parameters

    @cached_property
    def io_func(self):
        return ToParquetFunctionWrapper(
            self.engine,
            self.path,
            self.fs,
            self.partition_on,
            self.write_metadata_file,
            self.offset,
            self.name_function,
            self.write_kwargs,
        )

    def _divisions(self):
        return (None,) * (self.frame.npartitions + 1)

    def _task(self, index: int):
        return (self.io_func, (self.frame._name, index), (index,))


class ToParquetBarrier(Expr):
    _parameters = ToParquet._parameters

    @property
    def _meta(self):
        return None

    def _divisions(self):
        return (None, None)

    def _layer(self):
        if self.write_metadata_file:
            append = self.write_kwargs.get("append")
            compression = self.write_kwargs.get("compression")
            return {
                (self._name, 0): (
                    apply,
                    self.engine.write_metadata,
                    [
                        self.frame.__dask_keys__(),
                        self.fmd,
                        self.fs,
                        self.path,
                    ],
                    {"append": append, "compression": compression},
                )
            }
        else:
            return {(self._name, 0): (lambda x: None, self.frame.__dask_keys__())}


def to_parquet(
    df,
    path,
    compression="snappy",
    write_index=True,
    append=False,
    overwrite=False,
    ignore_divisions=False,
    partition_on=None,
    storage_options=None,
    custom_metadata=None,
    write_metadata_file=None,
    compute=True,
    compute_kwargs=None,
    schema="infer",
    name_function=None,
    filesystem=None,
    **kwargs,
):
    from dask_expr._collection import new_collection
    from dask_expr.io.parquet import NONE_LABEL, ToParquet

    engine = _set_parquet_engine(meta=df._meta)
    compute_kwargs = compute_kwargs or {}

    partition_on = partition_on or []
    if isinstance(partition_on, str):
        partition_on = [partition_on]

    if set(partition_on) - set(df.columns):
        raise ValueError(
            "Partitioning on non-existent column. "
            "partition_on=%s ."
            "columns=%s" % (str(partition_on), str(list(df.columns)))
        )

    if df.columns.inferred_type not in {"string", "empty"}:
        raise ValueError("parquet doesn't support non-string column names")

    if isinstance(engine, str):
        engine = get_engine(engine)

    if hasattr(path, "name"):
        path = stringify_path(path)

    fs, _paths, _, _ = engine.extract_filesystem(
        path,
        filesystem=filesystem,
        dataset_options={},
        open_file_options={},
        storage_options=storage_options,
    )
    assert len(_paths) == 1, "only one path"
    path = _paths[0]

    if overwrite:
        if append:
            raise ValueError("Cannot use both `overwrite=True` and `append=True`!")

        if fs.exists(path) and fs.isdir(path):
            # Check for any previous parquet ops reading from a file in the
            # output directory, since deleting those files now would result in
            # errors or incorrect results.
            for read_op in df.expr.find_operations(ReadParquet):
                read_path_with_slash = str(read_op.path).rstrip("/") + "/"
                write_path_with_slash = path.rstrip("/") + "/"
                if read_path_with_slash.startswith(write_path_with_slash):
                    raise ValueError(
                        "Cannot overwrite a path that you are reading "
                        "from in the same task graph."
                    )

            # Don't remove the directory if it's the current working directory
            if _is_local_fs(fs):
                working_dir = fs.expand_path(".")[0]
                if path.rstrip("/") == working_dir.rstrip("/"):
                    raise ValueError(
                        "Cannot clear the contents of the current working directory!"
                    )

            # It's safe to clear the output directory
            fs.rm(path, recursive=True)

        # Clear read_parquet caches in case we are
        # also reading from the overwritten path
        _cached_dataset_info.clear()
        _cached_plan.clear()

    # Always skip divisions checks if divisions are unknown
    if not df.known_divisions:
        ignore_divisions = True

    # Save divisions and corresponding index name. This is necessary,
    # because we may be resetting the index to write the file
    division_info = {"divisions": df.divisions, "name": df.index.name}
    if division_info["name"] is None:
        # As of 0.24.2, pandas will rename an index with name=None
        # when df.reset_index() is called.  The default name is "index",
        # but dask will always change the name to the NONE_LABEL constant
        if NONE_LABEL not in df.columns:
            division_info["name"] = NONE_LABEL
        elif write_index:
            raise ValueError(
                "Index must have a name if __null_dask_index__ is a column."
            )
        else:
            warnings.warn(
                "If read back by Dask, column named __null_dask_index__ "
                "will be set to the index (and renamed to None)."
            )

    # There are some "reserved" names that may be used as the default column
    # name after resetting the index. However, we don't want to treat it as
    # a "special" name if the string is already used as a "real" column name.
    reserved_names = []
    for name in ["index", "level_0"]:
        if name not in df.columns:
            reserved_names.append(name)

    # If write_index==True (default), reset the index and record the
    # name of the original index in `index_cols` (we will set the name
    # to the NONE_LABEL constant if it is originally `None`).
    # `fastparquet` will use `index_cols` to specify the index column(s)
    # in the metadata.  `pyarrow` will revert the `reset_index` call
    # below if `index_cols` is populated (because pyarrow will want to handle
    # index preservation itself).  For both engines, the column index
    # will be written to "pandas metadata" if write_index=True
    index_cols = []
    if write_index:
        real_cols = set(df.columns)
        none_index = list(df._meta.index.names) == [None]
        df = df.reset_index()
        if none_index:
            rename_columns = {c: NONE_LABEL for c in df.columns if c in reserved_names}
            df = df.rename(rename_columns)
        index_cols = [c for c in set(df.columns) - real_cols]
    else:
        # Not writing index - might as well drop it
        df = df.reset_index(drop=True)

    if custom_metadata and b"pandas" in custom_metadata.keys():
        raise ValueError(
            "User-defined key/value metadata (custom_metadata) can not "
            "contain a b'pandas' key.  This key is reserved by Pandas, "
            "and overwriting the corresponding value can render the "
            "entire dataset unreadable."
        )

    # Engine-specific initialization steps to write the dataset.
    # Possibly create parquet metadata, and load existing stuff if appending
    i_offset, fmd, metadata_file_exists, extra_write_kwargs = engine.initialize_write(
        df.to_dask_dataframe(),
        fs,
        path,
        append=append,
        ignore_divisions=ignore_divisions,
        partition_on=partition_on,
        division_info=division_info,
        index_cols=index_cols,
        schema=schema,
        custom_metadata=custom_metadata,
        **kwargs,
    )

    # By default we only write a metadata file when appending if one already
    # exists
    if append and write_metadata_file is None:
        write_metadata_file = metadata_file_exists

    # Check that custom name_function is valid,
    # and that it will produce unique names
    if name_function is not None:
        if not callable(name_function):
            raise ValueError("``name_function`` must be a callable with one argument.")
        filenames = [name_function(i + i_offset) for i in range(df.npartitions)]
        if len(set(filenames)) < len(filenames):
            raise ValueError("``name_function`` must produce unique filenames.")

    # If we are using a remote filesystem and retries is not set, bump it
    # to be more fault tolerant, as transient transport errors can occur.
    # The specific number 5 isn't hugely motivated: it's less than ten and more
    # than two.
    annotations = dask.config.get("annotations", {})
    if "retries" not in annotations and not _is_local_fs(fs):
        ctx = dask.annotate(retries=5)
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        out = new_collection(
            ToParquet(
                df.expr,
                path,
                fs,
                fmd,
                engine,
                i_offset,
                partition_on,
                write_metadata_file,
                name_function,
                toolz.merge(
                    kwargs,
                    {"compression": compression, "custom_metadata": custom_metadata},
                    extra_write_kwargs,
                ),
            )
        )

    if compute:
        out = out.compute(**compute_kwargs)

    # Invalidate the filesystem listing cache for the output path after write.
    # We do this before returning, even if `compute=False`. This helps ensure
    # that reading files that were just written succeeds.
    fs.invalidate_cache(path)

    return out


class ReadParquet(PartitionsFiltered, BlockwiseIO):
    """Read a parquet dataset"""

    _absorb_projections = True

    _parameters = [
        "path",
        "columns",
        "_partitions",
        "_series",
    ]
    _defaults = {
        "columns": None,
        # "parquet_file_extension": (".parq", ".parquet", ".pq"),
        "_partitions": None,
        "_series": False,
    }

    @functools.cached_property
    def _name(self):
        return "readparquet-" + _tokenize_deterministic(*self.operands)

    @functools.cached_property
    def filesystem(self):
        if str(self.path).startswith("s3://"):
            import boto3
            from pyarrow.fs import S3FileSystem

            bucket = self.path[5:].split("/")[0]
            session = boto3.session.Session()
            credentials = session.get_credentials()
            region = session.client("s3").get_bucket_location(Bucket=bucket)[
                "LocationConstraint"
            ]

            return S3FileSystem(
                secret_key=credentials.secret_key,
                access_key=credentials.access_key,
                region=region,
                session_token=credentials.token,
            )
        else:
            return None

    @property
    def columns(self):
        columns_operand = self.operand("columns")
        if columns_operand is None:
            return list(self._meta.columns)
        else:
            return _convert_to_list(columns_operand)

    @property
    def _meta(self):
        meta, _ = meta_and_filenames(self.path)
        if self.operand("columns") is not None:
            meta = meta[self.operand("columns")]
        if self._series:
            meta = meta[meta.columns[0]]
        return meta

    @functools.cached_property
    def _filename_batches(self):
        meta, filenames = meta_and_filenames(self.path)
        if not self.columns:
            files_per_partition = 1
        else:
            files_per_partition = int(round(len(meta.columns) / len(self.columns)))

        return list(toolz.partition_all(files_per_partition, filenames))

    def _filtered_task(self, i):
        batch = self._filename_batches[i]
        return (
            ReadParquet.to_pandas,
            (
                ReadParquet.read_partition,
                batch,
                self.columns,
                self.filesystem,
            ),
        )

    @staticmethod
    def to_pandas(t: pa.Table) -> pd.DataFrame:
        df = t.to_pandas(
            use_threads=False,
            ignore_metadata=False,
            types_mapper=types_mapper,
        )
        return df

    @staticmethod
    def read_partition(batch, columns, filesystem):
        def read_arrow_table(fn):
            t = pq.ParquetFile(fn, pre_buffer=True, filesystem=filesystem).read(
                columns=columns,
                use_threads=False,
                use_pandas_metadata=True,
            )
            return t

        if len(batch) == 1:
            return read_arrow_table(batch[0])
        if filesystem is None:  # local
            return pa.concat_tables(list(map(read_arrow_table, batch)))
        else:
            with concurrent.futures.ThreadPoolExecutor(len(batch)) as e:
                parts = list(e.map(read_arrow_table, batch))
            return pa.concat_tables(parts)

    def _divisions(self):
        meta, filenames = meta_and_filenames(self.path)
        files_per_partition = int(round(len(meta.columns) / len(self.columns)))
        return [None] * (int(math.ceil(len(filenames) / files_per_partition)) + 1)

    def _simplify_up(self, parent):
        if isinstance(parent, Index):
            # Column projection
            return self.substitute_parameters({"columns": [], "_series": False})

        if isinstance(parent, Projection):
            return BlockwiseIO._simplify_up(self, parent)

        # if isinstance(parent, Lengths):
        #     _lengths = self._get_lengths()
        #     if _lengths:
        #         return Literal(_lengths)

        # if isinstance(parent, Len):
        #     _lengths = self._get_lengths()
        #     if _lengths:
        #         return Literal(sum(_lengths))


def types_mapper(pyarrow_dtype):
    if pyarrow_dtype == pa.string():
        return pd.StringDtype("pyarrow")
    if "decimal" in str(pyarrow_dtype) or "date32" in str(pyarrow_dtype):
        return pd.ArrowDtype(pyarrow_dtype)


@functools.lru_cache
def meta_and_filenames(path):
    if str(path).startswith("s3://"):
        import s3fs

        filenames = s3fs.S3FileSystem().ls(path)

        import boto3
        from pyarrow.fs import S3FileSystem

        session = boto3.session.Session()
        credentials = session.get_credentials()

        bucket = path[5:].split("/")[0]
        region = session.client("s3").get_bucket_location(Bucket=bucket)[
            "LocationConstraint"
        ]

        filesystem = S3FileSystem(
            secret_key=credentials.secret_key,
            access_key=credentials.access_key,
            region=region,
            session_token=credentials.token,
        )
        path = path[5:]

    else:
        import glob
        import os

        if os.path.isdir(path):
            filenames = sorted(glob.glob(os.path.join(path, "*")))
        else:
            filenames = [path]  # TODO: split by row group

        filesystem = None

    ds = pq.ParquetDataset(path, filesystem=filesystem)
    t = pa.Table.from_pylist([], schema=ds.schema)
    meta = t.to_pandas(types_mapper=types_mapper)

    return meta, filenames


#
# Helper functions
#


def _set_parquet_engine(engine=None, meta=None):
    # Use `engine` or `meta` input to set the parquet engine
    if engine is None:
        if (
            meta is not None and typename(meta).split(".")[0] == "cudf"
        ) or dask.config.get("dataframe.backend", "pandas") == "cudf":
            from dask_cudf.io.parquet import CudfEngine

            engine = CudfEngine
        else:
            engine = "pyarrow"
    return engine


#
# Parquet-statistics handling
#


def _collect_pq_statistics(
    expr: ReadParquet, columns: list | None = None
) -> list[dict] | None:
    """Collect Parquet statistic for dataset paths"""

    # Be strict about columns argument
    if columns:
        if not isinstance(columns, list):
            raise ValueError(f"Expected columns to be a list, got {type(columns)}.")
        allowed = {expr._meta.index.name} | set(expr.columns)
        if not set(columns).issubset(allowed):
            raise ValueError(f"columns={columns} must be a subset of {allowed}")

    # Collect statistics using layer information
    fs = expr._io_func.fs
    parts = [
        part
        for i, part in enumerate(expr._plan["parts"])
        if not expr._filtered or i in expr._partitions
    ]

    # Execute with delayed for large and remote datasets
    parallel = int(False if _is_local_fs(fs) else 16)
    if parallel:
        # Group parts corresponding to the same file.
        # A single task should always parse statistics
        # for all these parts at once (since they will
        # all be in the same footer)
        groups = defaultdict(list)
        for part in parts:
            for p in [part] if isinstance(part, dict) else part:
                path = p.get("piece")[0]
                groups[path].append(p)
        group_keys = list(groups.keys())

        # Compute and return flattened result
        func = delayed(_read_partition_stats_group)
        result = dask.compute(
            [
                func(
                    list(
                        itertools.chain(
                            *[groups[k] for k in group_keys[i : i + parallel]]
                        )
                    ),
                    fs,
                    columns=columns,
                )
                for i in range(0, len(group_keys), parallel)
            ]
        )[0]
        return list(itertools.chain(*result))
    else:
        # Serial computation on client
        return _read_partition_stats_group(parts, fs, columns=columns)


def _read_partition_stats_group(parts, fs, columns=None):
    """Parse the statistics for a group of files"""

    def _read_partition_stats(part, fs, columns=None):
        # Helper function to read Parquet-metadata
        # statistics for a single partition

        if not isinstance(part, list):
            part = [part]

        column_stats = {}
        num_rows = 0
        columns = columns or []
        for p in part:
            piece = p["piece"]
            path = piece[0]
            row_groups = None if piece[1] == [None] else piece[1]
            with fs.open(path, default_cache="none") as f:
                md = pq.ParquetFile(f).metadata
            if row_groups is None:
                row_groups = list(range(md.num_row_groups))
            for rg in row_groups:
                row_group = md.row_group(rg)
                num_rows += row_group.num_rows
                for i in range(row_group.num_columns):
                    col = row_group.column(i)
                    name = col.path_in_schema
                    if name in columns:
                        if col.statistics and col.statistics.has_min_max:
                            if name in column_stats:
                                column_stats[name]["min"] = min(
                                    column_stats[name]["min"], col.statistics.min
                                )
                                column_stats[name]["max"] = max(
                                    column_stats[name]["max"], col.statistics.max
                                )
                            else:
                                column_stats[name] = {
                                    "min": col.statistics.min,
                                    "max": col.statistics.max,
                                }

        # Convert dict-of-dict to list-of-dict to be consistent
        # with current `dd.read_parquet` convention (for now)
        column_stats_list = [
            {
                "name": name,
                "min": column_stats[name]["min"],
                "max": column_stats[name]["max"],
            }
            for name in column_stats.keys()
        ]
        return {"num-rows": num_rows, "columns": column_stats_list}

    # Helper function used by _extract_statistics
    return [_read_partition_stats(part, fs, columns=columns) for part in parts]
