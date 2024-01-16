from __future__ import annotations

import pandas as pd
from dask.backends import CreationDispatch
from dask.dataframe.backends import DataFrameBackendEntrypoint

from dask_expr._dispatch import get_collection_type


class DXCreationDispatch(CreationDispatch):
    """Dask-expressions version of CreationDispatch"""

    # TODO Remove after https://github.com/dask/dask/pull/10794
    def dispatch(self, backend: str):
        from dask.backends import detect_entrypoints

        try:
            impl = self._lookup[backend]
        except KeyError:
            entrypoints = detect_entrypoints(f"dask-expr.{self._module_name}.backends")
            if backend in entrypoints:
                return self.register_backend(backend, entrypoints[backend].load()())
        else:
            return impl
        raise ValueError(f"No backend dispatch registered for {backend}")


dataframe_creation_dispatch = DXCreationDispatch(
    module_name="dataframe",
    default="pandas",
    entrypoint_class=DataFrameBackendEntrypoint,
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


@get_collection_type.register(pd.Series)
def get_collection_type_series(_):
    from dask_expr._collection import Series

    return Series


@get_collection_type.register(pd.DataFrame)
def get_collection_type_dataframe(_):
    from dask_expr._collection import DataFrame

    return DataFrame


@get_collection_type.register(pd.Index)
def get_collection_type_index(_):
    from dask_expr._collection import Index

    return Index


@get_collection_type.register(object)
def get_collection_type_object(_):
    from dask_expr._collection import Scalar

    return Scalar


######################################
# cuDF: Pandas Dataframes on the GPU #
######################################


@get_collection_type.register_lazy("cudf")
def _register_cudf():
    import dask_cudf  # noqa: F401
