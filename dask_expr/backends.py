from dask.base import normalize_token

# cuDF backend/dispatch utilities


@normalize_token.register_lazy("cudf")
def register_cudf():
    import cudf

    @normalize_token.register((cudf.DataFrame, cudf.Series, cudf.Index))
    def _tokenize_cudf_object(obj):
        return obj.iloc[:1].to_arrow()
