from dask.array.optimization import fuse_slice
from dask.array.slicing import normalize_slice, slice_array
from dask.array.utils import meta_from_array
from dask.utils import cached_property

from dask_expr.array.core import Array


class Slice(Array):
    _parameters = ["array", "index"]

    @property
    def _meta(self):
        return meta_from_array(self.array._meta, ndim=len(self.chunks))

    @cached_property
    def _info(self):
        return slice_array(
            self._name,
            self.array._name,
            self.array.chunks,
            self.index,
            self.array.dtype.itemsize,
        )

    def _layer(self):
        return self._info[0]

    @property
    def chunks(self):
        return self._info[1]

    def _simplify_down(self):
        if all(idx == slice(None, None, None) for idx in self.index):
            return self.array
        if isinstance(self.array, Slice):
            return Slice(
                self.array.array,
                normalize_slice(
                    fuse_slice(self.array.index, self.index), self.array.array.ndim
                ),
            )
