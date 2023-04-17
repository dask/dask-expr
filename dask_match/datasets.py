import functools

import numpy as np
import pandas as pd
from matchpy import CustomConstraint, Pattern, ReplacementRule, Wildcard

from dask.utils import random_state_data

from dask_match.collection import new_collection
from dask_match.expr import Projection
from dask_match.io import BlockwiseIO

__all__ = ["timeseries"]


class Timeseries(BlockwiseIO):
    _parameters = [
        "start",
        "end",
        "dtypes",
        "_projection",
        "freq",
        "partition_freq",
        "seed",
        "kwargs",
    ]
    _defaults = {
        "start": "2000-01-01",
        "end": "2000-12-31",
        "dtypes": None,
        "_projection": None,
        "freq": "10s",
        "partition_freq": "1M",
        "seed": None,
        "kwargs": {},
    }

    @property
    def projection(self):
        if self._projection is None:
            dtypes = self.operand("dtypes")
            return None if dtypes is None else list(dtypes.keys())
        return (
            self._projection
            if isinstance(self._projection, (list, pd.Index))
            else [self._projection]
        )

    @property
    def _meta(self):
        dtypes = self.operand("dtypes")
        return make_timeseries_part(
            "2000", "2000", dtypes, self.projection, "1H", 0, self.kwargs
        )

    def _divisions(self):
        return list(
            pd.date_range(start=self.start, end=self.end, freq=self.partition_freq)
        )

    @functools.cached_property
    def random_state(self):
        if self.seed is None:
            return np.random.randint(2e9, size=self.npartitions)
        else:
            return random_state_data(self.npartitions, self.seed)

    def _task(self, index):
        return (
            make_timeseries_part,
            self.divisions[index],
            self.divisions[index + 1],
            self.operand("dtypes"),
            self.projection,
            self.freq,
            self.random_state[index],
            self.kwargs,
        )

    @classmethod
    def _replacement_rules(self):
        start, end, dtypes, _projection, freq, partition_freq, seed, kwargs = map(
            Wildcard.dot,
            [
                "start",
                "end",
                "dtypes",
                "_projection",
                "freq",
                "partition_freq",
                "seed",
                "kwargs",
            ],
        )
        columns = Wildcard.dot("columns")

        def optimize_timeseries_projection(
            start, end, dtypes, _projection, freq, partition_freq, seed, kwargs, columns
        ):
            if isinstance(columns, (list, pd.Index)):
                return Timeseries(
                    start, end, dtypes, columns, freq, partition_freq, seed, kwargs
                )
            else:
                return Timeseries(
                    start, end, dtypes, columns, freq, partition_freq, seed, kwargs
                )[columns]

        def constraint(_projection, columns):
            """Avoid infinite loop with df["x"] -> df["x"]"""
            return isinstance(columns, (list, pd.Index)) or _projection is None

        yield ReplacementRule(
            Pattern(
                Projection(
                    Timeseries(
                        start,
                        end,
                        dtypes,
                        _projection,
                        freq,
                        partition_freq,
                        seed,
                        kwargs,
                    ),
                    columns,
                ),
                CustomConstraint(constraint),
            ),
            optimize_timeseries_projection,
        )


names = [
    "Alice",
    "Bob",
    "Charlie",
    "Dan",
    "Edith",
    "Frank",
    "George",
    "Hannah",
    "Ingrid",
    "Jerry",
    "Kevin",
    "Laura",
    "Michael",
    "Norbert",
    "Oliver",
    "Patricia",
    "Quinn",
    "Ray",
    "Sarah",
    "Tim",
    "Ursula",
    "Victor",
    "Wendy",
    "Xavier",
    "Yvonne",
    "Zelda",
]


def make_string(n, rstate):
    return rstate.choice(names, size=n)


def make_categorical(n, rstate):
    return pd.Categorical.from_codes(rstate.randint(0, len(names), size=n), names)


def make_float(n, rstate):
    return rstate.rand(n) * 2 - 1


def make_int(n, rstate, lam=1000):
    return rstate.poisson(lam, size=n)


make = {
    float: make_float,
    int: make_int,
    str: make_string,
    object: make_string,
    "string": make_string,
    "category": make_categorical,
}


def make_timeseries_part(start, end, dtypes, columns, freq, state_data, kwargs):
    index = pd.date_range(start=start, end=end, freq=freq, name="timestamp")
    state = np.random.RandomState(state_data)
    data = {}
    for k, dt in dtypes.items():
        kws = {
            kk.rsplit("_", 1)[1]: v
            for kk, v in kwargs.items()
            if kk.rsplit("_", 1)[0] == k
        }
        # Note: we compute data for all dtypes in order, not just those in the output
        # columns. This ensures the same output given the same state_data, regardless
        # of whether there is any column projection.
        # cf. https://github.com/dask/dask/pull/9538#issuecomment-1267461887
        result = make[dt](len(index), state, **kws)
        if k in columns:
            data[k] = result
    df = pd.DataFrame(data, index=index, columns=columns)
    if df.index[-1] == end:
        df = df.iloc[:-1]
    return df


def timeseries(
    start="2000-01-01",
    end="2000-01-31",
    freq="1s",
    partition_freq="1d",
    dtypes=None,
    seed=None,
    **kwargs,
):
    """Create timeseries dataframe with random data

    Parameters
    ----------
    start: datetime (or datetime-like string)
        Start of time series
    end: datetime (or datetime-like string)
        End of time series
    dtypes: dict (optional)
        Mapping of column names to types.
        Valid types include {float, int, str, 'category'}
    freq: string
        String like '2s' or '1H' or '12W' for the time series frequency
    partition_freq: string
        String like '1M' or '2Y' to divide the dataframe into partitions
    seed: int (optional)
        Randomstate seed
    kwargs:
        Keywords to pass down to individual column creation functions.
        Keywords should be prefixed by the column name and then an underscore.

    Examples
    --------
    >>> import dask_match.datasets import timeseries
    >>> df = timeseries(
    ...     start='2000', end='2010',
    ...     dtypes={'value': float, 'name': str, 'id': int},
    ...     freq='2H', partition_freq='1D', seed=1
    ... )
    >>> df.head()  # doctest: +SKIP
                           id      name     value
    2000-01-01 00:00:00   969     Jerry -0.309014
    2000-01-01 02:00:00  1010       Ray -0.760675
    2000-01-01 04:00:00  1016  Patricia -0.063261
    2000-01-01 06:00:00   960   Charlie  0.788245
    2000-01-01 08:00:00  1031     Kevin  0.466002
    """
    if dtypes is None:
        dtypes = {"name": "string", "id": int, "x": float, "y": float}

    if seed is None:
        seed = np.random.randint(2e9)

    projection = kwargs.pop("_projection", None)  # Not intended for public use
    expr = Timeseries(
        start, end, dtypes, projection, freq, partition_freq, seed, kwargs
    )
    return new_collection(expr)
