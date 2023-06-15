import functools

import pandas as pd
from dask.dataframe import methods
from dask.dataframe.dispatch import make_meta, meta_nonempty
from dask.dataframe.utils import check_meta, strip_unknown_categories
from dask.utils import apply, is_dataframe_like, is_series_like

from dask_expr.expr import AsType, Expr


class Concat(Expr):
    _parameters = ["frame", "join", "ignore_order", "_kwargs"]
    _defaults = {"join": "outer", "ignore_order": False, "_kwargs": {}}

    @functools.cached_property
    def _meta(self):
        meta = make_meta(
            methods.concat(
                [meta_nonempty(df._meta) for df in self.operand("frame")],
                join=self.join,
                filter_warning=False,
                **self._kwargs,
            )
        )
        return strip_unknown_categories(meta)

    def _divisions(self):
        dfs = self.operand("frame")
        if all(df.known_divisions for df in dfs):
            # each DataFrame's division must be greater than previous one
            if all(
                dfs[i].divisions[-1] < dfs[i + 1].divisions[0]
                for i in range(len(dfs) - 1)
            ):
                divisions = []
                for df in dfs[:-1]:
                    # remove last to concatenate with next
                    divisions += df.divisions[:-1]
                divisions += dfs[-1].divisions
                return divisions

        return [None] * (sum(df.npartitions for df in dfs) + 1)

    def _simplify_down(self):
        dfs = self.frame
        cast_dfs = []
        for df in dfs:
            # dtypes of all dfs need to be coherent
            # refer to https://github.com/dask/dask/issues/4685
            # and https://github.com/dask/dask/issues/5968.
            if is_dataframe_like(df.frame):
                shared_columns = df.columns.intersection(self._meta.columns)
                needs_astype = {
                    col: self._meta[col].dtype
                    for col in shared_columns
                    if df._meta[col].dtype != self._meta[col].dtype
                    and not isinstance(df[col].dtype, pd.CategoricalDtype)
                }

                if needs_astype:
                    cast_dfs.append(AsType(df.expr, dtypes=needs_astype))
                else:
                    cast_dfs.append(df)
            elif is_series_like(df) and is_series_like(self._meta):
                if not df.dtype == self._meta.dtype and not isinstance(
                    df.dtype, pd.CategoricalDtype
                ):
                    cast_dfs.append(AsType(df.expr, dtypes=self._meta.dtype))
                else:
                    cast_dfs.append(df)
            else:
                cast_dfs.append(df)
        x = StackPartition(
            cast_dfs,
            join=self.join,
            ignore_order=self.ignore_order,
            _kwargs=self._kwargs,
        )
        return x

    def _simplify_up(self, parent):
        return


class StackPartition(Concat):
    _parameters = ["frame", "join", "ignore_order", "_kwargs"]
    _defaults = {"join": "outer", "ignore_order": False, "_kwargs": {}}

    # @functools.cached_property
    # def _name(self):
    #     return funcname(type(self)).lower() + "-" + tokenize(*self.operand("frame"))

    def _layer(self):
        dsk, i = {}, 0
        for df in self.frame:
            try:
                check_meta(df._meta, self._meta)
                match = True
            except (ValueError, TypeError):
                match = False

            for key in df.__dask_keys__():
                if match:
                    dsk[(self._name, i)] = key
                else:
                    dsk[(self._name, i)] = (
                        apply,
                        methods.concat,
                        [[self._meta, key], 0, self.join, False, True],
                        self._kwargs,
                    )
                i += 1
        return dsk

    def _simplify_down(self):
        return
