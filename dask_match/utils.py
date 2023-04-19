def _check_culling(expr, partitions):
    # Test utility to check culling.
    # Checks that "_partitions" is set to the
    # expected value for all expressions defining
    # a "_partitions" parameter
    for dep in expr.dependencies():
        _check_culling(dep, partitions)
    if "_partitions" in expr._parameters:
        assert expr._partitions == partitions
    return
