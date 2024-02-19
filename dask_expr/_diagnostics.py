def explain(expr, fuse: bool = True):
    import graphviz

    from dask_expr._expr import Expr, optimize_blockwise_fusion

    g = graphviz.Digraph("g", filename=f"explain-{expr._name}")
    g.node_attr.update(shape="record")

    def generate_stage(expr: Expr, name: str, graph: graphviz.Digraph) -> str:
        seen = set(expr._name)
        stack = [expr]

        subgraph_id = f"cluster-{name}"
        with graph.subgraph(name=subgraph_id) as c:
            c.attr(label=name)
            while stack:
                node = stack.pop()
                c.node(f"{subgraph_id}-{node._name}", label=node._explain_label())

                for dep in node.dependencies():
                    c.edge(f"{subgraph_id}-{dep._name}", f"{subgraph_id}-{node._name}")
                    if dep._name not in seen:
                        seen.add(dep._name)
                        stack.append(dep)
        return subgraph_id

    result = expr
    generate_stage(result, "Logical Plan", g)
    # Simplify
    result = result.simplify()
    generate_stage(result, "Simplified Logical Plan", g)

    # Manipulate Expression to make it more efficient
    result = result.rewrite(kind="tune")
    generate_stage(result, "Tuned Logical Plan", g)

    # Lower
    result = result.lower_completely()
    generate_stage(result, "Physical Plan", g)

    # Simplify again
    result = result.simplify()
    generate_stage(result, "Simplified Physical Plan", g)

    # Final graph-specific optimizations
    if fuse:
        result = optimize_blockwise_fusion(result)
        generate_stage(result, "Fused Physical Plan", g)

    g.view()
