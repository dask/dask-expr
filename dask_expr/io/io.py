from __future__ import annotations

import functools
import math

from dask.dataframe.core import is_dataframe_like
from dask.dataframe.io.io import sorted_division_locations

from dask_expr._expr import (
    Blockwise,
    Expr,
    Lengths,
    Literal,
    PartitionsFiltered,
    Projection,
)
from dask_expr._reductions import Len
from dask_expr._util import _convert_to_list


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
        return dict(self.operand("layer"))


class BlockwiseIO(Blockwise, IO):
    _absorb_projections = False

    def _simplify_up(self, parent):
        if (
            self._absorb_projections
            and isinstance(parent, Projection)
            and is_dataframe_like(self._meta)
        ):
            # Column projection
            parent_columns = parent.operand("columns")
            substitutions = {"columns": _convert_to_list(parent_columns)}
            if isinstance(parent_columns, (str, int)):
                substitutions["_series"] = True
            return self.substitute_parameters(substitutions)

    def _combine_similar(self, root: Expr):
        if self._absorb_projections:
            # For ReadParquet, we can avoid redundant file-system
            # access by aggregating multiple operations with different
            # column projections into the same operation.
            alike = self._find_similar_operations(root, ignore=["columns", "_series"])
            if alike:
                # We have other ReadParquet operations in the expression
                # graph that can be combined with this one.

                # Find the column-projection union needed to combine
                # the qualified ReadParquet operations
                columns_operand = self.operand("columns")
                if columns_operand is None:
                    columns_operand = self.columns
                columns = set(columns_operand)
                rps = [self] + alike
                for rp in alike:
                    rp_columns = rp.operand("columns")
                    if rp_columns is None:
                        rp_columns = rp.columns
                    columns |= set(rp_columns)
                columns = sorted(columns)

                # Can bail if we are not changing columns or the "_series" operand
                if columns_operand == columns and (
                    len(columns) > 1 or not self._series
                ):
                    return

                # Check if we have the operation we want elsewhere in the graph
                for rp in rps:
                    # if rp.operand("columns") == columns and not rp.operand("_series"):
                    if rp.columns == columns and not rp.operand("_series"):
                        return (
                            rp[columns_operand[0]]
                            if self._series
                            else rp[columns_operand]
                        )

                # Create the "combined" ReadParquet operation
                subs = {"columns": columns}
                if self._series:
                    subs["_series"] = False
                new = self.substitute_parameters(subs)
                return new[columns_operand[0]] if self._series else new[columns_operand]

        return


class FromPandas(PartitionsFiltered, BlockwiseIO):
    """The only way today to get a real dataframe"""

    _parameters = ["frame", "npartitions", "sort", "columns", "_partitions", "_series"]
    _defaults = {
        "npartitions": 1,
        "sort": True,
        "columns": None,
        "_partitions": None,
        "_series": False,
    }
    _pd_length_stats = None
    _absorb_projections = True

    @property
    def _meta(self):
        meta = self.frame.head(0)
        if self.columns:
            return meta[self.columns[0]] if self._series else meta[self.columns]
        return meta

    @functools.cached_property
    def columns(self):
        columns_operand = self.operand("columns")
        if columns_operand is None:
            try:
                return list(self.frame.columns)
            except AttributeError:
                return []
        else:
            return _convert_to_list(columns_operand)

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

    def _get_lengths(self) -> tuple | None:
        if self._pd_length_stats is None:
            locations = self._locations()
            self._pd_length_stats = tuple(
                offset - locations[i]
                for i, offset in enumerate(locations[1:])
                if not self._filtered or i in self._partitions
            )
        return self._pd_length_stats

    def _simplify_up(self, parent):
        if isinstance(parent, Lengths):
            _lengths = self._get_lengths()
            if _lengths:
                return Literal(_lengths)

        if isinstance(parent, Len):
            _lengths = self._get_lengths()
            if _lengths:
                return Literal(sum(_lengths))

        if isinstance(parent, Projection):
            return super()._simplify_up(parent)

    def _divisions(self):
        return self._divisions_and_locations[0]

    def _locations(self):
        return self._divisions_and_locations[1]

    def _filtered_task(self, index: int):
        start, stop = self._locations()[index : index + 2]
        part = self.frame.iloc[start:stop]
        if self.columns:
            return part[self.columns[0]] if self._series else part[self.columns]
        return part

    def __str__(self):
        return "df"

    __repr__ = __str__
