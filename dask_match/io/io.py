import functools
import math

from dask.base import tokenize

from dask_match.expr import Blockwise, BlockwiseInput, Expr


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


class FromPandas(BlockwiseIO):
    """The only way today to get a real dataframe"""

    _parameters = ["frame", "npartitions"]
    _defaults = {"npartitions": 1}

    @functools.cached_property
    def _name(self):
        return "from-pandas-" + tokenize(self.frame, self.npartitions)

    @property
    def _meta(self):
        return self.frame.head(0)

    def _divisions(self):
        return [None] * (self.npartitions + 1)

    def _parts(self):
        partsize = int(math.ceil(len(self.frame) / self.npartitions))
        locations = list(range(0, len(self.frame), partsize)) + [len(self.frame)]
        return [
            self.frame.iloc[start:stop]
            for start, stop in zip(locations[:-1], locations[1:])
        ]

    @functools.cached_property
    def _blockwise_input(self):
        return BlockwiseInput(self._parts())

    def dependencies(self):
        return [self._blockwise_input]

    def _blockwise_task(self, index: int | None = None):
        if index is None:
            return self._blockwise_input._name
        return self._blockwise_input[index]

    def __str__(self):
        return "df"

    __repr__ = __str__
