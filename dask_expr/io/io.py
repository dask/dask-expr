import functools
import math

from dask.dataframe.io.io import sorted_division_locations

from dask_expr.expr import Blockwise, Expr, PartitionsFiltered


class IO(Expr):
    def __str__(self):
        return f"{type(self).__name__}({self._name[-7:]})"


class FromGraph(IO):
    """A DataFrame created from an opaque Dask task graph

    This is used in persist, for example, and would also be used in any
    conversion from legacy dataframes.
    """

    _parameters = ["layer", "_meta", "divisions", "_name"]

    @property
    def _meta(self):
        return self.operand("_meta")

    def _divisions(self):
        return self.operand("divisions")

    @property
    def _name(self):
        return self.operand("_name")

    def _layer(self):
        return self.operand("layer")


class BlockwiseIO(Blockwise, IO):
    pass


class FromPandas(PartitionsFiltered, BlockwiseIO):
    """The only way today to get a real dataframe"""

    _parameters = ["frame", "npartitions", "sort", "_partitions"]
    _defaults = {"npartitions": 1, "sort": True, "_partitions": None}

    @property
    def _meta(self):
        return self.frame.head(0)

    @functools.cached_property
    def _divisions_and_locations(self):
        data = self.frame
        nrows = len(data)
        npartitions = self.operand("npartitions")
        if self.sort:
            if not data.index.is_monotonic_increasing:
                data = data.sort_index(ascending=True)
            divisions, locations = sorted_division_locations(
                data.index,
                npartitions=npartitions,
                chunksize=None,
            )
        else:
            chunksize = int(math.ceil(nrows / npartitions))
            locations = list(range(0, nrows, chunksize)) + [len(data)]
            divisions = (None,) * len(locations)
        return divisions, locations

    def _lengths(self, force: bool = False) -> tuple | None:
        locations = self._locations()
        return tuple(
            offset - locations[i]
            for i, offset in enumerate(locations[1:])
            if not self._filtered or i in self._partitions
        )

    def _divisions(self):
        return self._divisions_and_locations[0]

    def _locations(self):
        return self._divisions_and_locations[1]

    def _filtered_task(self, index: int):
        start, stop = self._locations()[index : index + 2]
        return self.frame.iloc[start:stop]

    def __str__(self):
        return "df"

    __repr__ = __str__
