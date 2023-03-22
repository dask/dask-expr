import functools

from dask_match.collection.core import new_collection
from dask_match.expr.io.io import IO


class ReadCSV(IO):
    _parameters = ["filename", "usecols", "header"]
    _defaults = {"usecols": None, "header": "infer"}

    @functools.cached_property
    def _ddf(self):
        # Temporary hack to simplify logic
        import dask.dataframe as dd

        return dd.read_csv(
            self.filename,
            usecols=self.usecols,
            header=self.header,
        )

    @property
    def _meta(self):
        return self._ddf._meta

    def _divisions(self):
        return self._ddf.divisions

    def _layer(self):
        return self._ddf.dask.to_dict()


def read_csv(*args, **kwargs):
    return new_collection(ReadCSV(*args, **kwargs))