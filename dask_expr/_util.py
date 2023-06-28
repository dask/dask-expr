from __future__ import annotations


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


class TaggedDispatch:
    """Tagged single dispatch."""

    def __init__(self, name=None):
        self._lookup = {}
        if name:
            self.__name__ = name

    def register(self, type, tag: str = "general", func=None):
        """Register dispatch of `func` on arguments of type `type`"""

        def wrapper(func):
            if isinstance(type, tuple):
                for t in type:
                    self.register((t, tag), func)
            else:
                self._lookup[(type, tag)] = func
            return func

        return wrapper(func) if func is not None else wrapper

    def dispatch(self, cls, tag="general"):
        """Return the function implementation for the given ``cls``"""
        lk = self._lookup
        for cls2 in cls.__mro__:
            try:
                impl = lk[(cls2, tag)]
            except KeyError:
                pass
            else:
                if cls is not cls2:
                    # Cache lookup
                    lk[(cls, tag)] = impl
                return impl
        raise TypeError(f"No dispatch for {(cls, tag)}")

    def __call__(self, arg, *args, tag: str = "general", **kwargs):
        """
        Call the corresponding method based on type of argument.
        """
        meth = self.dispatch(type(arg), tag)
        return meth(arg, *args, **kwargs)

    @property
    def __doc__(self):
        try:
            func = self.dispatch(object)
            return func.__doc__
        except TypeError:
            return "Tagged Dispatch for %s" % self.__name__


simplify_down_dispatch = TaggedDispatch("simplify_down")
simplify_up_dispatch = TaggedDispatch("simplify_up")
