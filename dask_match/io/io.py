import math
from functools import cached_property

from dask_match.core import Expr, output_partitions_rule


class IO(Expr):
    pass


class FromPandas(IO):
    """The only way today to get a real dataframe"""

    _parameters = ["frame", "npartitions", "output_partitions"]
    _defaults = {"npartitions": 1, "output_partitions": None}

    @classmethod
    def _replacement_rules(cls):
        # Output-partition projection
        yield output_partitions_rule(cls)

    @property
    def _meta(self):
        return self.frame.head(0)

    def _divisions(self):
        return [None] * (len(self._chunks) + 1)

    @cached_property
    def _chunks(self):
        chunksize = int(math.ceil(len(self.frame) / self.npartitions))
        locations = list(range(0, len(self.frame), chunksize)) + [len(self.frame)]
        chunks = [
            self.frame.iloc[start:stop]
            for start, stop in zip(locations[:-1], locations[1:])
        ]
        if self.output_partitions is None:
            return chunks
        else:
            return [chunks[i] for i in self.output_partitions]

    def _layer(self):
        return {(self._name, i): chunk for i, chunk in enumerate(self._chunks)}

    def __str__(self):
        return "df"

    __repr__ = __str__


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
