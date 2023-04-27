import functools

from dask.dataframe.dispatch import make_meta, meta_nonempty
from dask.utils import apply

from dask_expr.expr import Blockwise, Expr
from dask_expr.io import FromPandas
from dask_expr.repartition import Repartition
from dask_expr.shuffle import Shuffle, _contains_index_name


class Merge(Expr):
    """Abstract merge operation"""

    _parameters = [
        "left",
        "right",
        "how",
        "on",
        "left_on",
        "right_on",
        "left_index",
        "right_index",
        "suffixes",
        "indicator",
        "backend",
        "shuffle_backend",
        "_partitions",
    ]
    _defaults = {
        "how": "inner",
        "on": None,
        "left_on": None,
        "right_on": None,
        "left_index": False,
        "right_index": False,
        "suffixes": ("_x", "_y"),
        "indicator": False,
        "backend": None,
        "shuffle_backend": None,
        "_partitions": None,
    }

    def __str__(self):
        return f"Merge({self._name[-7:]})"

    @property
    def _kwargs(self):
        return {
            k: self.operand(k)
            for k in [
                "how",
                "on",
                "left_on",
                "right_on",
                "left_index",
                "right_index",
                "suffixes",
                "indicator",
            ]
        }

    @functools.cached_property
    def _meta(self):
        left = self._as_meta(self.left)
        right = self._as_meta(self.right)
        return left.merge(right, **self._kwargs)

    def _divisions(self):
        npartitions_left = self._as_expr(self.left).npartitions
        npartitions_right = self._as_expr(self.right).npartitions
        npartitions = max(npartitions_left, npartitions_right)
        return (None,) * (npartitions + 1)

    @staticmethod
    def _as_meta(dep, nonempty=True):
        if isinstance(dep, Expr):
            dep = dep._meta
        return meta_nonempty(dep) if nonempty else make_meta(dep)

    @staticmethod
    def _as_expr(dep):
        if isinstance(dep, Expr):
            return dep
        return FromPandas(dep, 1)

    def _simplify_down(self):
        # Lower from an abstract expression using
        # logic in MergeBackend.from_abstract_merge
        backend = self.backend or MergeBackend
        if hasattr(backend, "from_abstract_merge"):
            return backend.from_abstract_merge(self)
        else:
            raise ValueError(f"{backend} not supported")


class MergeBackend(Merge):
    """Base merge-backend class"""

    def _simplify_down(self):
        return None

    @classmethod
    def from_abstract_merge(cls, expr: Merge) -> Expr:
        """Return a new Exprs to perform a merge"""

        # TODO:
        #  1. Handle mixed indexed merge
        #  2. Add multi-partition broadcast merge
        #  3. Add/leverage partition statistics

        left = expr._as_expr(expr.left)
        right = expr._as_expr(expr.right)
        how = expr.how
        on = expr.on
        left_on = expr.left_on
        right_on = expr.right_on
        left_index = expr.left_index
        right_index = expr.right_index

        for o in [on, left_on, right_on]:
            if isinstance(o, Expr):
                raise NotImplementedError()
        if (
            not on
            and not left_on
            and not right_on
            and not left_index
            and not right_index
        ):
            on = [c for c in left.columns if c in right.columns]
            if not on:
                left_index = right_index = True

        if on and not left_on and not right_on:
            left_on = right_on = on
            on = None

        supported_how = ("left", "right", "outer", "inner")
        if how not in supported_how:
            raise ValueError(
                f"dask.dataframe.merge does not support how='{how}'."
                f"Options are: {supported_how}."
            )

        # Check for "trivial" broadcast (single partition)
        npartitions = max(left.npartitions, right.npartitions)
        if (
            npartitions == 1
            or left.npartitions == 1
            and how in ("right", "inner")
            or right.npartitions == 1
            and how in ("left", "inner")
        ):
            return BlockwiseMerge(left, right, **expr._kwargs)

        # Check if we are merging on indices with known divisions
        merge_indexed_left = (
            left_index or _contains_index_name(left, left_on)
        ) and left.known_divisions
        merge_indexed_right = (
            right_index or _contains_index_name(right, right_on)
        ) and right.known_divisions

        shuffle_left_on = left_on
        shuffle_right_on = right_on
        if merge_indexed_left and merge_indexed_right:
            # fully-indexed merge
            if left.npartitions >= right.npartitions:
                right = Repartition(right, new_divisions=left.divisions, force=True)
            else:
                left = Repartition(left, new_divisions=right.divisions, force=True)
            shuffle_left_on = shuffle_right_on = None
        # TODO: Need 'rearrange_by_divisions' equivalent
        # to avoid shuffle when we are merging on known
        # divisions on one side only.
        elif left_index:
            shuffle_left_on = left.index._meta.name
            if shuffle_left_on is None:
                raise NotImplementedError()
        elif right_index:
            shuffle_right_on = right.index._meta.name
            if shuffle_right_on is None:
                raise NotImplementedError()

        if shuffle_left_on:
            # Shuffle left
            left = Shuffle(
                left,
                shuffle_left_on,
                npartitions_out=npartitions,
                backend=expr.shuffle_backend,
            )

        if shuffle_right_on:
            # Shuffle right
            right = Shuffle(
                right,
                shuffle_right_on,
                npartitions_out=npartitions,
                backend=expr.shuffle_backend,
            )

        # Blockwise merge
        return BlockwiseMerge(left, right, **expr._kwargs)


class BlockwiseMerge(MergeBackend, Blockwise):
    """Base merge-backend class"""

    def _broadcast_dep(self, dep: Expr):
        return dep.npartitions == 1

    @staticmethod
    def _merge_partition(df, other, **kwargs):
        return df.merge(other, **kwargs)

    def _task(self, index: int):
        return (
            apply,
            self._merge_partition,
            [
                self._blockwise_arg(self.left, index),
                self._blockwise_arg(self.right, index),
            ],
            self._kwargs,
        )
