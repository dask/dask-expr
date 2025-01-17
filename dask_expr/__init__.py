import warnings

warnings.warn(
    "Dask-expr is now part of the main dask package. Please remove dask-expr from your environment.",
    DeprecationWarning,
)

__version__ = "v2.0.0"
