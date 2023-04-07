import functools

from dask.base import tokenize

from dask_match.core import MappedArg
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

    @functools.lru_cache
    def _subgraph_dependencies(self):
        # Need to pass `token` to ensure deterministic name
        return [MappedArg([t[1] for t in self._tasks], token=tokenize(self._ddf))]

    def _blockwise_subgraph(self):
        dsk = self._tasks[0][0].dsk
        return {
            self._name: (
                next(iter(dsk.values()))[0],
                self._subgraph_dependencies()[0]._name,
            )
        }
