from __future__ import annotations

from collections.abc import Sequence


def _convert_to_list(column) -> list | None:
    if column is None or isinstance(column, list):
        pass
    elif isinstance(column, tuple):
        column = list(column)
    elif hasattr(column, "dtype"):
        column = column.tolist()
    else:
        column = [column]
    return column


def is_scalar(x):
    return not (isinstance(x, Sequence) or hasattr(x, "dtype"))
