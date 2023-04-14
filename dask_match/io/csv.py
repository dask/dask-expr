import functools

from dask.base import tokenize

from dask_match.expr import BlockwiseInput
from dask_match.io.io import BlockwiseIO


class ReadCSV(BlockwiseIO):
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

    @property
    def _tasks(self):
        return list(self._ddf.dask.to_dict().values())

    @functools.cached_property
    def _indexable_input(self) -> dict:
        name = f"csvdep-{tokenize(self._ddf)}"
        return {name: [t[1] for t in self._tasks]}

    @functools.lru_cache
    def dependencies(self):
        name = f"csvdep-{tokenize(self._ddf)}"
        return [BlockwiseInput([t[1] for t in self._tasks], name=name)]

    @functools.cached_property
    def _io_func(self):
        dsk = self._tasks[0][0].dsk
        return next(iter(dsk.values()))[0]

    def _blockwise_task(self, index: int | None = None):
        #dep = list(self._indexable_input)[0]
        dep = self.dependencies()[0]
        return (self._io_func, self._blockwise_arg(dep, index))
