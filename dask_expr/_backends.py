from __future__ import annotations

import pandas as pd
from dask.backends import CreationDispatch
from dask.dataframe.backends import DataFrameBackendEntrypoint

dataframe_creation_dispatch = CreationDispatch(
    module_name="dataframe",
    default="pandas",
    entrypoint_class=DataFrameBackendEntrypoint,
    entrypoint_root="dask_expr",  # Differs from `dask.dataframe`
    name="dataframe_creation_dispatch",
)


class PandasBackendEntrypoint(DataFrameBackendEntrypoint):
    """Pandas-Backend Entrypoint Class for Dask-Expressions

    Note that all DataFrame-creation functions are defined
    and registered 'in-place'.
    """

    @classmethod
    def to_backend_dispatch(cls):
        from dask.dataframe.dispatch import to_pandas_dispatch

        return to_pandas_dispatch

    @classmethod
    def to_backend(cls, data, **kwargs):
        if isinstance(data._meta, (pd.DataFrame, pd.Series, pd.Index)):
            # Already a pandas-backed collection
            return data
        return data.map_partitions(cls.to_backend_dispatch(), **kwargs)


dataframe_creation_dispatch.register_backend("pandas", PandasBackendEntrypoint())
