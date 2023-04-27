import functools

from dask.dataframe.dispatch import make_meta, meta_nonempty
from dask.utils import apply

from dask_expr.expr import Blockwise, Expr
from dask_expr.io import FromPandas
from dask_expr.shuffle import Shuffle


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
        #  1. Handle merge on index
        #  2. Add multi-partition broadcast merge
        #  3. Add/leverage partition statistics

        left = expr._as_expr(expr.left)
        right = expr._as_expr(expr.right)
        npartitions = max(left.npartitions, right.npartitions)
        how = expr.how

        # Check for "trivial" broadcast (single partition)
        simple_broadcast = (
            npartitions == 1
            or left.npartitions == 1
            and how in ("right", "inner")
            or right.npartitions == 1
            and how in ("left", "inner")
        )

        if not simple_broadcast:
            if expr.left_index or expr.right_index:
                # TODO: Merge on index
                raise NotImplementedError()

            left_on = expr.left_on or expr.on
            right_on = expr.right_on or expr.on

            # Shuffle left & right
            left = Shuffle(
                left,
                left_on,
                npartitions_out=npartitions,
                backend=expr.shuffle_backend,
            )
            right = Shuffle(
                right,
                right_on,
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
