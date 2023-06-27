import functools

from tlz import merge_sorted, unique

from dask_expr.expr import Expr, Projection
from dask_expr.repartition import RepartitionDivisions


class AlignDivisions(Expr):
    @property
    def _meta(self):
        return self._frame._meta

    @property
    def _frame(self):
        return self.operands[0]

    @functools.cached_property
    def dfs(self):
        _is_broadcastable = functools.partial(is_broadcastable, self.dependencies())
        return [
            df
            for df in self.dependencies()
            if df.ndim > 0 and not _is_broadcastable(df)
        ]

    def _divisions(self):
        divisions = list(unique(merge_sorted(*[df.divisions for df in self.dfs])))
        if len(divisions) == 1:  # single value for index
            divisions = (divisions[0], divisions[0])
        return divisions

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            return type(self)(
                self._frame[parent.operand("columns")], *self.operands[1:]
            )

    def _lower(self):
        if not self.dfs:
            return self._frame

        if all(df.divisions == self.dfs[0].divisions for df in self.dfs):
            return self._frame

        if not all(df.known_divisions for df in self.dfs):
            raise ValueError(
                "Not all divisions are known, can't align "
                "partitions. Please use `set_index` "
                "to set the index."
            )

        return RepartitionDivisions(
            self._frame, new_divisions=self.divisions, force=True
        )


def is_broadcastable(dfs, s):
    """
    This Series is broadcastable against another dataframe in the sequence
    """

    def compare(s, df):
        try:
            return s.divisions == (min(df.columns), max(df.columns))
        except TypeError:
            return False

    return (
        s.ndim <= 1
        and s.npartitions == 1
        and s.known_divisions
        and any(compare(s, df) for df in dfs if df.ndim == 2)
    )
