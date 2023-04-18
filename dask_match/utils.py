def _check_take_partitions(expr, partitions):
    # Test utility to check culling.
    # Checks that "_take_partitions" is set to the
    # expected value for all expressions defining
    # a "_take_partitions" parameter
    for dep in expr.dependencies():
        _check_take_partitions(dep, partitions)
    if "_take_partitions" in expr._parameters:
        assert expr._take_partitions == partitions
    return
