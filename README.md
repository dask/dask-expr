Dask Expressions
================

Dask Dataframes with query optimization.

This is a proof-of-concept rewrite of Dask dataframe that includes query
optimization and generally improved organization.

Example
-------

```python
import dask_expr as dx

df = dx.datasets.timeseries()
df.head()

df.groupby("name").x.mean().compute()
```

Query Representation
--------------------

Dask-expr encodes user code in an expression tree:

```python
>>> df.x.mean().pprint()

Mean:
  Projection: columns='x'
    Timeseries: seed=1896674884
```

This expression tree will be optimized and modified before execution:

```python
>>> df.x.mean().optimize().pprint()

Div:
  Sum:
    Fused(375f9):
    | Projection: columns='x'
    |   Timeseries: dtypes={'x': <class 'float'>} seed=1896674884
  Count:
    Fused(375f9):
    | Projection: columns='x'
    |   Timeseries: dtypes={'x': <class 'float'>} seed=1896674884
```

Stability
---------

This project is a work in progress and will be changed without notice or
deprecation warning.  Please provide feedback, but it's best to avoid use in
production settings.


API Coverage
------------

**`dask_expr.DataFrame`**

- `abs`
- `align`
- `all`
- `any`
- `apply`
- `assign`
- `astype`
- `clip`
- `combine_first`
- `copy`
- `count`
- `dask`
- `drop`
- `drop_duplicates`
- `dropna`
- `eval`
- `explode`
- `fillna`
- `groupby`
- `head`
- `idxmax`
- `idxmin`
- `index`
- `isin`
- `isna`
- `join`
- `map`
- `map_partitions`
- `max`
- `memory_usage`
- `merge`
- `min`
- `min`
- `mode`
- `nlargest`
- `nsmallest`
- `nunique_approx`
- `partitions`
- `prod`
- `rename`
- `rename_axis`
- `repartition`
- `replace`
- `reset_index`
- `round`
- `sample`
- `select_types`
- `set_index`
- `shuffle`
- `sum`
- `tail`
- `to_parquet`
- `to_timestamp`
- `visualize`


**`dask_expr.Series`**

- `abs`
- `align`
- `all`
- `any`
- `apply`
- `astype`
- `between`
- `clip`
- `combine_first`
- `copy`
- `count`
- `dask`
- `drop_duplicates`
- `dropna`
- `explode`
- `fillna`
- `groupby`
- `head`
- `idxmax`
- `idxmin`
- `index`
- `isin`
- `isna`
- `map`
- `map_partitions`
- `max`
- `memory_usage`
- `min`
- `min`
- `mode`
- `nlargest`
- `nsmallest`
- `nunique_approx`
- `partitions`
- `prod`
- `rename_axis`
- `repartition`
- `replace`
- `reset_index`
- `round`
- `shuffle`
- `sum`
- `tail`
- `to_frame`
- `to_timestamp`
- `unique`
- `value_counts`
- `visualize`


**`dask_expr.Index`**

- `abs`
- `align`
- `all`
- `any`
- `apply`
- `astype`
- `clip`
- `combine_first`
- `copy`
- `count`
- `dask`
- `fillna`
- `groupby`
- `head`
- `idxmax`
- `idxmin`
- `index`
- `isin`
- `isna`
- `map_partitions`
- `max`
- `memory_usage`
- `min`
- `min`
- `mode`
- `nunique_approx`
- `partitions`
- `prod`
- `rename_axis`
- `repartition`
- `replace`
- `reset_index`
- `round`
- `shuffle`
- `sum`
- `tail`
- `to_frame`
- `to_timestamp`
- `visualize`


**`dask_expr._groupby.GroupBy`**

- `agg`
- `aggregate`
- `count`
- `first`
- `last`
- `max`
- `mean`
- `min`
- `prod`
- `size`
- `std`
- `sum`
- `value_counts`
- `var`
