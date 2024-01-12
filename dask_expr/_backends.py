from __future__ import annotations

import pandas as pd
from dask.backends import CreationDispatch, detect_entrypoints
from dask.dataframe.backends import DataFrameBackendEntrypoint


class DaskExprCreationDispatch(CreationDispatch):
    """Dask-Expr version of CreationDispatch

    TODO: This code can all go away if CreationDispatch
    makes it possible to override the entrypoint path.
    We just want to allow external libraries to expose
    a dask-expr entrypoint and dask (legacy) entrypoint
    at the same time.
    """

    def detect_entrypoints(self):
        return detect_entrypoints(f"dask-expr.{self._module_name}.backends")

    def dispatch(self, backend: str):
        """Return the desired backend entrypoint"""
        try:
            impl = self._lookup[backend]
        except KeyError:
            # Check entrypoints for the specified backend
            entrypoints = self.detect_entrypoints()
            if backend in entrypoints:
                return self.register_backend(backend, entrypoints[backend].load()())
        else:
            return impl
        raise ValueError(f"No backend dispatch registered for {backend}")


dataframe_creation_dispatch = DaskExprCreationDispatch(
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
