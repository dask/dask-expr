def _maybe_convert_to_list(column):
    if (
        column is not None
        and not isinstance(column, list)
        and not hasattr(column, "dtype")
    ):
        column = [column]
    return column
