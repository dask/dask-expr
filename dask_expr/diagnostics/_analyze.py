import functools
from typing import Any

from dask.base import DaskMethodsMixin
from dask.utils import format_bytes

from dask_expr._expr import Blockwise, Expr
from dask_expr.io.io import FusedIO


def inject_analyze(expr: Expr, id: str, injected: dict) -> Expr:
    if expr._name in injected:
        return injected[expr._name]

    new_operands = []
    for operand in expr.operands:
        if isinstance(operand, Expr) and not isinstance(expr, FusedIO):
            new = inject_analyze(operand, id, injected)
            injected[operand._name] = new
        else:
            new = operand
        new_operands.append(new)
    return Analyze(type(expr)(*new_operands), id, expr._name)


from dask_expr.diagnostics._explain import _add_graphviz_edges, _explain_info


def analyze(expr: Expr, format: str | None, **kwargs: Any):
    import graphviz
    from distributed import get_client, wait

    from dask_expr import new_collection
    from dask_expr.diagnostics._analyze_plugin import AnalyzePlugin

    client = get_client()

    client.register_plugin(AnalyzePlugin())

    # TODO: Make this work with fuse=True
    expr = expr.optimize(fuse=False)

    analysis_id = expr._name

    # Inject analyze nodes
    injected = inject_analyze(expr, analysis_id, {})
    out = new_collection(injected)
    _ = DaskMethodsMixin.compute(out, **kwargs)
    wait(_)

    # Collect data
    statistics = client.sync(client.scheduler.analyze_get_statistics, id=analysis_id)

    # Plot statistics in graph
    seen = set(expr._name)
    stack = [expr]

    if format is None:
        format = "png"

    g = graphviz.Digraph("g", filename=f"analyze-{expr._name}", format="png")
    g.node_attr.update(shape="record")
    while stack:
        node = stack.pop()
        info = _explain_info(node)
        info = _analyze_info(node, statistics[node._name])
        _add_graphviz_node(info, g)
        _add_graphviz_edges(info, g)

        if isinstance(node, FusedIO):
            continue
        for dep in node.operands:
            if not isinstance(dep, Expr) or dep._name in seen:
                continue
            seen.add(dep._name)
            stack.append(dep)

    g.view()


def _add_graphviz_node(info, graph):
    nbytes = info["cost"]["nbytes"]
    label = "".join(
        [
            "<{<b>",
            info["label"],
            "</b> | ",
            "<br />".join(
                [f"{key}: {value}" for key, value in info["details"].items()]
            ),
            " | ",
            "nbytes:<br />",
            ", ".join(
                [f"{key}: {format_bytes(value)}" for key, value in nbytes.items()]
            ),
            "}>",
        ]
    )

    graph.node(info["name"], label)


from dask_expr.diagnostics._explain import _explain_info


def _analyze_info(expr: Expr, digest):
    info = _explain_info(expr)
    info["cost"] = {
        "nbytes": {
            "min": digest.min(),
            "median": digest.quantile(0.5),
            "max": digest.max(),
        }
    }
    return info


def analyze_operation(frame, analysis_id, expr_name):
    from dask.sizeof import sizeof

    from dask_expr.diagnostics._analyze_plugin import get_worker_plugin

    worker_plugin = get_worker_plugin()
    worker_plugin.add(analysis_id, expr_name, sizeof(frame))
    return frame


class Analyze(Blockwise):
    _parameters = ["frame", "analysis_id", "expr_name"]

    operation = staticmethod(analyze_operation)

    @functools.cached_property
    def _meta(self):
        return self.frame._meta
