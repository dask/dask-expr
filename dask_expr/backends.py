from dask.base import normalize_token

# cuDF backend/dispatch utilities
# TODO: cuDF-specific dispatching functions and
# utilities should eventually live outside of
# dask-expr (probably).
#
# WARNING: Everything in this file should be considered
# private and experimental!


@normalize_token.register_lazy("cudf")
def register_cudf():
    import cudf

    @normalize_token.register((cudf.DataFrame, cudf.Series, cudf.Index))
    def _tokenize_cudf_object(obj):
        # Convert cudf objects to arrow to ensure
        # deterministic tokenization
        return obj.iloc[:1].to_arrow()


def _cudf_parquet_engine():
    # Temporary utility to return patched version
    # of `dask_cudf.io.parquet.CudfEngine` for
    # dask-expr testing
    from dask_cudf.io.parquet import CudfEngine

    class PatchedCudfEngine(CudfEngine):
        @classmethod
        def _create_dd_meta(cls, *args, **kwargs):
            import cudf

            meta = CudfEngine._create_dd_meta(*args, **kwargs)
            return cudf.from_pandas(meta)

    return PatchedCudfEngine
