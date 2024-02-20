from dask_expr._expr import Blockwise


def analyze(expr, fuse: bool = True):
    # Install plugins

    # Inject analyze nodes

    # Collect data

    # plot
    pass


class Analyze(Blockwise):
    _parameters = ["frame"]

    @staticmethod
    def operation(frame):
        pass
