from dask_expr import _version, datasets
from dask_expr._collection import *

__version__ = _version.get_versions()["version"]

from dask.dataframe._compat import PANDAS_GE_200

if not PANDAS_GE_200:
    warnings.warn(
        "The installed Pandas version is not recommended for "
        "`dask_expr`. Please update to `pandas>=2` if possible."
        "\nRAPIDS users can ignore this warning."
    )
